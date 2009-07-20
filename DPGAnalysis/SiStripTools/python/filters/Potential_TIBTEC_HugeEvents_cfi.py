import FWCore.ParameterSet.Config as cms

potentialTIBTECHugeEvents = cms.EDFilter('EventWithHistoryEDFilter',
                             partitionName = cms.untracked.string("TI"),
                             absBXInCycleRangeLtcyAware = cms.untracked.vint32(8,8)
                             )
