import FWCore.ParameterSet.Config as cms

try:
    from hcalLaserBadEvents_2011 import badEvents
except:
    print "<hcalLaserEventFilter_cfi> WARNING!  No explicit list 'badEvents.py' containing bad HCAL laser run/events was found!"
    badEvents=[]

hcalLaserEventFilter = cms.EDFilter("HcalLaserEventFilter",
                                    vetoByRunEventNumber=cms.untracked.bool(False),
                                    # Add each bad event as run #, event # in this list
                                    BadRunEventNumbers=cms.untracked.vuint32(badEvents
                                                                             # badEvents file contains 218 bad events, as of 28 Nov. 2011

    ),
                                    vetoByHBHEOccupancy=cms.untracked.bool(True),
                                    minOccupiedHBHE=cms.untracked.uint32(4000), # minimum number of HBHErechits that must be present for HBHEOccupancy filter to remove event
                                    
                                    debug = cms.untracked.bool(False),
                                    reverseFilter = cms.untracked.bool(False), # if True, will select only events failing filter, rather than events passing
                                    hbheInputLabel=cms.untracked.InputTag("hbhereco"),
                                    hcalNoiseSummaryLabel=cms.untracked.InputTag("hcalnoise"),
                                    taggingMode   = cms.bool(False),
                                    maxerrormessage = cms.untracked.int32(5), # max number of error messages to print
                                    forceUseRecHitCollection=cms.untracked.bool(False),  # if specified, will attempt to use HBHE RecHit Collection directly; otherwise, will use information as stored in HcalNoiseSummary object for CMSSW_5_2_0 and above.  (If CMSSW version is < 5_2_0, RecHit collection will be used automatically, since HcalNoiseSummary in those versions didn't contain this filter info)
                                    forceUseHcalNoiseSummary=cms.untracked.bool(False),  # Can likewise specify to force the use of Hcal Noise Summary, regardless of CMSSW version.   Perhaps this should be the default, since version checked doesn't allow for comparison of patched older versions with new version?
                                    )
