import FWCore.ParameterSet.Config as cms

potentialTOBFrameHeaderEventsSPeak = cms.EDFilter('EventWithHistoryEDFilter',
                                                commonConfiguration = cms.untracked.PSet(
                                                           partitionName = cms.untracked.string("TO")
                                                ),
                                            filterConfigurations = cms.untracked.VPSet(
                                               cms.PSet(
                                                  absBXInCycleRangeLtcyAware = cms.untracked.vint32(38,40)
                                               )
                                            )
                                         )
# foo bar baz
# Y6hMeGNj8njAJ
