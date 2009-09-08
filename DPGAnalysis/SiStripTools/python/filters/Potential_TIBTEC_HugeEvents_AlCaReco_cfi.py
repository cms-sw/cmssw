import FWCore.ParameterSet.Config as cms

PotentialTIBTECHugeEvents = cms.EDFilter('EventWithHistoryEDFilter',
                             partitionName              = cms.untracked.string("TM"),
                             absBXInCycleRangeLtcyAware = cms.untracked.vint32(8,11),
                             historyProduct             = cms.untracked.InputTag("ConsecutiveHEs"),
                             APVPhaseLabel              = cms.untracked.string("apvPhases")
                             )
