#! /usr/bin/env python
import ROOT
from DataFormats.FWLite import Events, Handle
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ('analysis')
options.parseArguments()
events = Events(options)

print "In total there are %d events" % events.size()
print "Trying an event loop"
for i,event in enumerate(events):
    print "I am processing an event"
    if i > 10: break
print "Done with the event loops"
