#!/usr/bin/env python

#Quotas {'Datatype':{'FDSN':[[runt_type,keep_period,min_num_lumis]]}}
# Any run/dataset stays the longest time from all applicable quotas.

# Collisions runs as old as 2 years.
# Cosmic runs that lasted 50 lumisections or more on the past 6 months
# Cosmic runs that lasted 25 lumisections or more on the past 3 months
# Cosmic runs that lasted less than 25 lumisections from the last month
# Test runs from the last month
# By default things stay in the GUI for a year
{'online_data':{'.*':[
  [COLLISIONS_RUN,365*2,0],
  [COSMICS_RUN,6 * 30,50],
  [COSMICS_RUN,3 * 30,25],
  [COSMICS_RUN,30,0],
  [TEST_RUN,1,0],
  [TEST_RUN,15,25],
  [TEST_RUN,30,100]
  ]}}
