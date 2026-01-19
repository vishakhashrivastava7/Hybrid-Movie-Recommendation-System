import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# ----------------------------
# Step 1: Load Data
# ----------------------------
movies = pd.read_csv('movies.csv')  # movieId, title, genres
ratings = pd.read_csv('u.data', sep='\t', names=['userId','movieId','rating','timestamp'])
tags = pd.read_csv('tags.csv')      # userId, movieId, tag, timestamp

# ----------------------------
# Step 2: Data Cleaning
# ----------------------------
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
tags['timestamp'] = pd.to_datetime(tags['timestamp'], unit='s')
movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)
tags.drop_duplicates(inplace=True)

# ----------------------------
# Step 3: Collaborative Filtering (Item-Item)
# ----------------------------
user_movie_matrix = ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
item_similarity = cosine_similarity(user_movie_matrix)
item_sim_df = pd.DataFrame(item_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
scaler = MinMaxScaler()
item_sim_scaled = pd.DataFrame(scaler.fit_transform(item_sim_df), index=item_sim_df.index, columns=item_sim_df.columns)

# ----------------------------
# Step 4: Content-Based Filtering (Genres)
# ----------------------------
genres_df = movies['genres'].str.get_dummies('|')
content_sim = cosine_similarity(genres_df)
content_sim_df = pd.DataFrame(content_sim, index=movies['movieId'], columns=movies['movieId'])
content_sim_scaled = pd.DataFrame(scaler.fit_transform(content_sim_df), index=content_sim_df.index, columns=content_sim_df.columns)

# ----------------------------
# Step 5: Tag-Based Similarity
# ----------------------------
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: " ".join(x)).reset_index()
movie_tags = pd.merge(movie_tags, movies[['movieId','title']], on='movieId', how='left')

tfidf = TfidfVectorizer(stop_words='english')
tag_matrix = tfidf.fit_transform(movie_tags['tag'])
tag_sim = cosine_similarity(tag_matrix)
tag_sim_df = pd.DataFrame(tag_sim, index=movie_tags['movieId'], columns=movie_tags['movieId'])
tag_scaled = pd.DataFrame(scaler.fit_transform(tag_sim_df), index=tag_sim_df.index, columns=tag_sim_df.columns)

# ----------------------------
# Step 6: Combine All Similarities (Hybrid)
# ----------------------------
item_sim_scaled = item_sim_scaled.reindex(index=tag_scaled.index, columns=tag_scaled.columns, fill_value=0)
content_sim_scaled = content_sim_scaled.reindex(index=tag_scaled.index, columns=tag_scaled.columns, fill_value=0)

# Weights: 50% collaborative, 25% content, 25% tags
hybrid_sim = 0.5*item_sim_scaled + 0.25*content_sim_scaled + 0.25*tag_scaled

# ----------------------------
# Step 7: Diversity Hybrid Recommendation Function
# ----------------------------
def recommend_movies_diverse(movie_title, top_n=10, year_filter=None):
    try:
        movie_id = movies[movies['title'].str.contains(movie_title, case=False, regex=False)]['movieId'].values[0]
    except IndexError:
        return f"Movie '{movie_title}' not found in dataset."

    # Top 50 candidates for diversity
    similar_scores = hybrid_sim[movie_id].sort_values(ascending=False)
    similar_scores = similar_scores.drop(movie_id)
    top_candidates = similar_scores.head(top_n*5).index  # 5x candidates

    recommendations = movies[movies['movieId'].isin(top_candidates)][['movieId','title','genres']].copy()

    # Optional release year filter
    if year_filter:
        recommendations = recommendations[recommendations['title'].str.contains(str(year_filter))]

    # Pick random N movies for diversity
    if len(recommendations) > top_n:
        recommendations = recommendations.sample(n=top_n, random_state=42)

    return recommendations

# ----------------------------
# Step 8: Streamlit Dashboard
# ----------------------------
st.title("🎬 Portfolio-Ready Movie Recommendation System (No Posters)")
st.write("Hybrid engine: Ratings + Genres + Tags | Diverse Top Recommendations")

movie_input = st.text_input("Enter a movie title:")
num_rec = st.slider("Number of recommendations", 1, 15, 10)
year_filter = st.text_input("Optional: Filter by release year (YYYY)")

if movie_input:
    recs = recommend_movies_diverse(movie_input, top_n=num_rec, year_filter=year_filter if year_filter else None)
    if isinstance(recs, str):
        st.warning(recs)
    else:
        for idx, row in recs.iterrows():
            st.subheader(row['title'])
            st.write("Genres:", row['genres'])
