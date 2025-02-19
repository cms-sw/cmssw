import FWCore.ParameterSet.Config as cms

TOBTickmarksEvents = cms.EDFilter('EventWithHistoryEDFilter',
                                                commonConfiguration = cms.untracked.PSet(
                                                           partitionName = cms.untracked.string("TO")
                                                ),
                             filterConfigurations = cms.untracked.VPSet(
                                cms.PSet(
                                     absBXInCycleRangeLtcyAware = cms.untracked.vint32(16,19)
                                )
                             )
                          )
