import FWCore.ParameterSet.Config as cms

PotentialTIBTECFrameHeaderEventsAdditionalPeak = cms.EDFilter('EventWithHistoryEDFilter',
                                                     partitionName              = cms.untracked.string("TI"),
                                                     absBXInCycleRangeLtcyAware = cms.untracked.vint32(24,26),
                                                     historyProduct             = cms.untracked.InputTag("ConsecutiveHEs"),
                                                     APVPhaseLabel              = cms.untracked.string("apvPhases")
                                                 )
