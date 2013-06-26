import FWCore.ParameterSet.Config as cms

tibHugeEvents = cms.EDFilter('EventWithHistoryEDFilter',
                                                commonConfiguration = cms.untracked.PSet(
                                                           partitionName = cms.untracked.string("TI")
                                                ),
                             filterConfigurations = cms.untracked.VPSet(
                                cms.PSet(
                                    dbxInCycleRangeLtcyAware = cms.untracked.vint32(77,77)
                                )
                             )
                    )
