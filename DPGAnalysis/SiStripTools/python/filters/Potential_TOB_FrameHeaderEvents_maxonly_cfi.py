import FWCore.ParameterSet.Config as cms

potentialTOBFrameHeaderEventsMax = cms.EDFilter('EventWithHistoryEDFilter',
                                                partitionName = cms.untracked.string("TO"),
                                                absBXInCycleRangeLtcyAware = cms.untracked.vint32(19,19)
                                                 )
