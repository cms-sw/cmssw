#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot


## General syntax to import specific functions in a library: 
##from (library) import (specific library function)
from pandas import DataFrame, read_csv

# General syntax to import a library but no functions: 
##import (library) as (give the library a nickname/alias)
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)

# The inital set of baby names and bith rates
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]

BabyDataSet = list(zip(names,births))
print BabyDataSet

df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])
print df

df.to_csv('births1880.csv',index=False,header=False)

Location = './births1880.csv'
df = pd.read_csv(Location)

print df

df = pd.read_csv(Location, names=['Names','Births'])
print df

print df.dtypes

# Check data type of Births column
print df.Births.dtype

# Method 1:
Sorted = df.sort_values(['Births'], ascending=False)
print Sorted.head(1)

# Method 2:
print df['Births'].max()

# Create graph
df['Births'].plot()

# Maximum value in the data set
MaxValue = df['Births'].max()

# Name associated with the maximum value
MaxName = df['Names'][df['Births'] == df['Births'].max()].values

# Text to display on graph
Text = str(MaxValue) + " - " + MaxName

# Add text to graph
pyplot.annotate(Text, xy=(1, MaxValue), xytext=(8, 0), 
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

print("The most popular name")
print df[df['Births'] == df['Births'].max()]



