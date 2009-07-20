import FWCore.ParameterSet.Config as cms

potentialTOBHugeEvents = cms.EDFilter('EventWithHistoryEDFilter',
                             partitionName = cms.untracked.string("TO"),
                             absBXInCycleRangeLtcyAware = cms.untracked.vint32(8,8)
                             )
