import FWCore.ParameterSet.Config as cms

tibHugeEvents = cms.EDFilter('EventWithHistoryEDFilter',
                             partitionName = cms.untracked.string("TI"),
                             dbxInCycleRangeLtcyAware = cms.untracked.vint32(77,77)
                             )
