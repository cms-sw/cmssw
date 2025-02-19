import FWCore.ParameterSet.Config as cms

tobFrameHeaderEvents = cms.EDFilter('EventWithHistoryEDFilter',
                                                commonConfiguration = cms.untracked.PSet(
                                                           partitionName = cms.untracked.string("TO")
                                                ),
                               filterConfigurations = cms.untracked.VPSet(
                                  cms.PSet(
                                    dbxInCycleRangeLtcyAware = cms.untracked.vint32(298,319)
                                  )
                               )
                           )
