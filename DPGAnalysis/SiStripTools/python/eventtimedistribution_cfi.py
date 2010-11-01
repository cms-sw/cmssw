import FWCore.ParameterSet.Config as cms

eventtimedistribution = cms.EDAnalyzer('EventTimeDistribution',
                                      historyProduct = cms.InputTag("consecutiveHEs"),
                                      apvPhaseCollection = cms.InputTag("APVPhases"),
                                      phasePartition = cms.untracked.string("All"),
                                      minBinSizeInSec = cms.untracked.double(1.) 
)	
