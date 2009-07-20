import FWCore.ParameterSet.Config as cms

TOBTickmarksEvents = cms.EDFilter('EventWithHistoryEDFilter',
                                     partitionName = cms.untracked.string("TO"),
                                     absBXInCycleRangeLtcyAware = cms.untracked.vint32(16,19)
                                     )
