import FWCore.ParameterSet.Config as cms

tibtecFrameHeaderEvents = cms.EDFilter('EventWithHistoryEDFilter',
                                                commonConfiguration = cms.untracked.PSet(
                                                           partitionName = cms.untracked.string("TI")
                                                ),
                                  filterConfigurations = cms.untracked.VPSet(
                                     cms.PSet(
                                       dbxInCycleRangeLtcyAware = cms.untracked.vint32(298,319)
                                     )
                                  )
                              )
