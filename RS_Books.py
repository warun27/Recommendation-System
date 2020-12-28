# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 05:37:28 2020

@author: shara
"""

import pandas as pd
import numpy as np
books = pd.read_csv("G:/DS Assignments/Recommendation_system/book.csv",encoding=('latin-1'))
books.shape
books.head()
books["Book.Rating"]
books["Book.Rating"].isnull().sum()

from sklearn.metrics.pairwise import linear_kernel
cosine_sin_matrix = linear_kernel(books[["Book.Rating"]], books[["Book.Rating"]])
print(cosine_sin_matrix)
books_index = pd.Series(books.index, index = books['Book.Title']).drop_duplicates()
books_index["Jane Doe"]

def get_books_recomendations(Name,topN):
    books_id = books_index[Name]
    cosine_scores = list(enumerate(cosine_sin_matrix[books_id]))
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    cosine_scores_10 = cosine_scores[0:topN + 1]
    books_id = [i[0] for i in cosine_scores_10]
    books_scores = [i[1] for i in cosine_scores_10] 
    books_similar = pd.DataFrame(columns = ["name", "score"])
    books_similar['name'] = books.loc[books_id, "Book.Title"]
    books_similar['score'] = books_scores
    books_similar.reset_index(inplace = True)  
    books_similar.drop(["index"],axis=1,inplace=True)
    print(books_similar)
    
get_books_recomendations("The Prince", 20)


