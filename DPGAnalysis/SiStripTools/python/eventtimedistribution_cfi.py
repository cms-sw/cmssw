import FWCore.ParameterSet.Config as cms

eventtimedistribution = cms.EDAnalyzer('EventTimeDistribution',
                                      historyProduct = cms.InputTag("consecutiveHEs"),
                                      apvPhaseCollection = cms.InputTag("APVPhases"),
                                      minBinSizeInSec = cms.untracked.double(1.) 
)	
