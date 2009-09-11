#! /usr/bin/env python

from DataFormats.FWLite import Events, Handle
import optparse

print "starting python test"

files = ['good.root', 'good_delta5.root']
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

print "Python test succeeded!"

