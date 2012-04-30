#! /usr/bin/env python

from DataFormats.FWLite import Events, Handle
import optparse

print "starting python test"

files = ['empty_a.root', 'good_a.root', 'empty_a.root', 'good_b.root']
events = Events (files)

thingHandle = Handle ('std::vector<edmtest::Thing>')
indicies = events.fileIndicies()
for event in events:
    newIndicies = event.fileIndicies()
    if indicies != newIndicies:
        print "new file"
    indicies = newIndicies
    event.getByLabel ('Thing', thingHandle)
    thing = thingHandle.product()
    for loop in range (thing.size()):
        print thing.at (loop).a

events.toBegin()

for event in events:
    pass

events.toBegin()

for event in events:
    event.getByLabel ('Thing', thingHandle)
    thing = thingHandle.product()
    for loop in range (thing.size()):
        print thing.at (loop).a

for i in xrange(events.size()):
    if not events.to(i):
        print "failed to go to index ",i
        exit(1)

print "Python test succeeded!"

