import FWCore.ParameterSet.Config as cms

potentialTIBTECFrameHeaderEventsFPeak = cms.EDFilter('EventWithHistoryEDFilter',
                                                     partitionName = cms.untracked.string("TI"),
                                                     absBXInCycleRangeLtcyAware = cms.untracked.vint32(19,22)
                                                 )
