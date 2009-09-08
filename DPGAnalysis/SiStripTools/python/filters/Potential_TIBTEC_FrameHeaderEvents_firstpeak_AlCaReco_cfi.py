import FWCore.ParameterSet.Config as cms

PotentialTIBTECFrameHeaderEventsFPeak = cms.EDFilter('EventWithHistoryEDFilter',
                                                     partitionName              = cms.untracked.string("TM"),
                                                     absBXInCycleRangeLtcyAware = cms.untracked.vint32(19,22),
                                                     historyProduct             = cms.untracked.InputTag("ConsecutiveHEs"),
                                                     APVPhaseLabel              = cms.untracked.string("apvPhases")
                                                 )
