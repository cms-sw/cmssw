import FWCore.ParameterSet.Config as cms

potentialTIBTECFrameHeaderEventsMax = cms.EDFilter('EventWithHistoryEDFilter',
                                                commonConfiguration = cms.untracked.PSet(
                                                           partitionName = cms.untracked.string("TI")
                                                ),
                                               filterConfigurations = cms.untracked.VPSet(
                                                 cms.PSet(
                                                   absBXInCycleRangeLtcyAware = cms.untracked.vint32(19,19)
                                                 )
                                               )
                                             )
