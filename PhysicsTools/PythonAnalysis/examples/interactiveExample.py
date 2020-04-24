# has to be called with python -i interactiveExample.py

from PhysicsTools.PythonAnalysis import *
from ROOT import *

# prepare the FWLite autoloading mechanism
gSystem.Load("libFWCoreFWLite.so")
FWLiteEnabler::enable()

# enable support for files > 2 GB
gSystem.Load("libIOPoolTFileAdaptor")
ui = TFileAdaptorUI()


# load the example file from castor
theFile = TFile.Open("castor:/castor/cern.ch/cms/store/CSA06/CSA06-106-os-Jets-0/AOD/CMSSW_1_0_6-AODSIM-H15a59ba7b4c3d9e291172f60a399301f/1025/96C3197B-0264-DB11-9A9C-00304885AD72.root")


# access the event tree
print "=============================="
print "Loading event tree"
events = EventTree(theFile)

print "Start looping over some events"
for event in events:
      photons = event.photons
      print "  Number of photons in event %i: %i" % (event, len(photons))
      if event > 2: break  # workaround will become obsolete 



#####################################################
# all following commands have been used interactively
#
## accessing photons 
# print photon[0]
#
## looping over the photons. what's there?
#for photon in photons:
#  print photon.energy() 
#
## selecting photons
#selectedPhotons = [p for p in photons if p.energy()> 3]
#
## looking at the results
#for photon in selectedPhotons:
#  print photon.energy()
#
## how to find out about aliases
#for alias in events.getListOfAliases():
#  print alias
#
## how to learn about an object
#help(photon[0])  
#
## how to leave the session
#
# Ctrl-D

