#!/usr/bin/env python
#
# Example of how to analyse HLT objects using FWLite and pyROOT.
# 

# adapted from PhysicsTools/PatExamples/bin/PatBasicFWLiteAnalyzer.py

import ROOT
import sys
from DataFormats.FWLite import Events, Handle

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------


events = Events (["input.root"])
handle = Handle ("trigger::TriggerFilterObjectWithRefs")

# declare a variable 'electrons' to get them from the trigger objects
ROOT.gROOT.ProcessLine("std::vector<reco::ElectronRef> electrons;")

# copied from DataFormats/HLTReco/interface/TriggerTypeDefs.h
TriggerElectron       = +82

# note that for this collection there is only one (not two for L1Iso and L1NonIso)
label = ("hltL1NonIsoHLTNonIsoSingleElectronEt22TighterEleIdOneOEMinusOneOPFilter")

numElectronsSeen = 0
numEventsWithElectron = 0

# loop over events
for event in events:
    # use getByLabel, just like in cmsRun
    event.getByLabel (label, handle)
    # get the product
    trigobjs = handle.product()

    trigobjs.getObjects(TriggerElectron, ROOT.electrons)
    print "number of electrons in this event:",len(ROOT.electrons)

    bestOneOverEminusOneOverP = None

    numElectronsSeen += len(ROOT.electrons)

    if len(ROOT.electrons) > 0:
        numEventsWithElectron += 1
    else:
        continue

    for eleindex, electron in enumerate(ROOT.electrons):

        print "electron",eleindex
        # see HLTrigger/Egamma/src/HLTElectronOneOEMinusOneOPFilterRegional.cc
        # how 1/E-1/p is calculated 
        tracks = electron.track().product()

        superClusters = electron.superCluster().product()

        print "  number of tracks:",len(tracks)
        print "  number of superclusters:",len(superClusters)

        for track in tracks:
            momentum = track.p()

            for superCluster in superClusters:
                energy = superCluster.energy()

                thisOneOverEminusOneOverP = abs(1/energy - 1/momentum)
                
                print "    momentum=",momentum,"energy=",energy,"E/P=",energy/momentum,"1/E-1/p=",thisOneOverEminusOneOverP


                if bestOneOverEminusOneOverP == None or thisOneOverEminusOneOverP < bestOneOverEminusOneOverP:
                    bestOneOverEminusOneOverP = thisOneOverEminusOneOverP

            # loop over clusters

        # loop over tracks

    # loop over electron trigger objects

    print "best value:",bestOneOverEminusOneOverP

    

print "total number of electrons:",numElectronsSeen
print "events with at least one electron:",numEventsWithElectron
