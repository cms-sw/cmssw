#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot

#First histogram tutorial in histogrammar package
import histogrammar as hg

# generate a stream of uniform random numbers
import random
data = [random.random() for i in xrange(2000)]

# aggregation structure and fill rule
histogram = hg.Bin(num=20, low=0, high=1, quantity=lambda x: x, value=hg.Count())

# fill the histogram!
for d in data:
    histogram.fill(d)

# quick plotting convenience method using matplotlib (if the user has this installed)
ax = histogram.plot.matplotlib(name="hello world!")

pyplot.savefig('histogrammar.png')

