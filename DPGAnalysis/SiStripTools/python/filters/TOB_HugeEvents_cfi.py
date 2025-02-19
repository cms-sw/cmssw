import FWCore.ParameterSet.Config as cms

tobHugeEvents = cms.EDFilter('EventWithHistoryEDFilter',
                                                commonConfiguration = cms.untracked.PSet(
                                                           partitionName = cms.untracked.string("TO")
                                                ),
                        filterConfigurations = cms.untracked.VPSet(
                          cms.PSet(
                             dbxInCycleRangeLtcyAware = cms.untracked.vint32(77,77)
                          )
                        )
                    )
