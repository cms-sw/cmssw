import FWCore.ParameterSet.Config as cms

potentialTOBFrameHeaderEventsMax = cms.EDFilter('EventWithHistoryEDFilter',
                                                commonConfiguration = cms.untracked.PSet(
                                                           partitionName = cms.untracked.string("TO")
                                                ),
                                          filterConfigurations = cms.untracked.VPSet(
                                             cms.PSet(
                                                absBXInCycleRangeLtcyAware = cms.untracked.vint32(19,19)
                                             )
                                          )
                                       )
# foo bar baz
# 8RWIB4EDmWEg9
