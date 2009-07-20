import FWCore.ParameterSet.Config as cms

tobHugeEvents = cms.EDFilter('EventWithHistoryEDFilter',
                             partitionName = cms.untracked.string("TO"),
                             dbxInCycleRangeLtcyAware = cms.untracked.vint32(77,77)
                             )
