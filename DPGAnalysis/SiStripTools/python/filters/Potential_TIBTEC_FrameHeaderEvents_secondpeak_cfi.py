import FWCore.ParameterSet.Config as cms

potentialTIBTECFrameHeaderEventsSPeak = cms.EDFilter('EventWithHistoryEDFilter',
                                                     partitionName = cms.untracked.string("TI"),
                                                     absBXInCycleRangeLtcyAware = cms.untracked.vint32(38,40)
                                                 )
