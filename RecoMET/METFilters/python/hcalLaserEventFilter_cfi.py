import FWCore.ParameterSet.Config as cms

try:
    from hcalLaserBadEvents_2011 import badEvents
except:
    print "<hcalLaserEventFilter_cfi> WARNING!  No explicit list 'badEvents.py' containing bad HCAL laser run/events was found!"
    badEvents=[]

hcalLaserEventFilter = cms.EDFilter("HcalLaserEventFilter",
                                    vetoByRunEventNumber=cms.untracked.bool(True),
                                    # Add each bad event as run #, event # in this list
                                    BadRunEventNumbers=cms.untracked.vuint32(badEvents
                                                                             # badEvents file contains 218 bad events, as of 28 Nov. 2011

    ),
                                    vetoByHBHEOccupancy=cms.untracked.bool(False),
                                    minOccupiedHBHE=cms.untracked.uint32(5000), # minimum number of HBHErechits that must be present for HBHEOccupancy filter to remove event
                                    
                                    debug = cms.untracked.bool(False),
                                    reverseFilter = cms.untracked.bool(False), # if True, will select only events failing filter, rather than events passing
                                    hbheInputLabel=cms.untracked.InputTag("hbhereco"),

                                    taggingMode   = cms.bool(False),
                                    )
