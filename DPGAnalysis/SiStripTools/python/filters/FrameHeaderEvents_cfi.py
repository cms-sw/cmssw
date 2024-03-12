import FWCore.ParameterSet.Config as cms

frameHeaderEvents = cms.EDFilter('EventWithHistoryEDFilter',
                                 commonConfiguration = cms.untracked.PSet(
                                            partitionName = cms.untracked.string("Any")
                                                ),
                                  filterConfigurations = cms.untracked.VPSet(
                                     cms.PSet(
                                       dbxInCycleRangeLtcyAware = cms.untracked.vint32(298,319)
                                     )
                                  )
                              )
# foo bar baz
# g7GAdwdbhpfvu
# RY5E2K2pBJvhn
