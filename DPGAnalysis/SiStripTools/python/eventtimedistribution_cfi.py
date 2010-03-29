import FWCore.ParameterSet.Config as cms

eventtimedistribution = cms.EDAnalyzer('EventTimeDistribution',
                                      historyProduct = cms.InputTag("consecutiveHEs"),
)	
