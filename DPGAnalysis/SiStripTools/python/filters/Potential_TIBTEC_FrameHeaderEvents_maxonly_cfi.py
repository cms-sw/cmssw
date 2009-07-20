import FWCore.ParameterSet.Config as cms

potentialTIBTECFrameHeaderEventsMax = cms.EDFilter('EventWithHistoryEDFilter',
                                                   partitionName = cms.untracked.string("TI"),
                                                   absBXInCycleRangeLtcyAware = cms.untracked.vint32(19,19)
                                                 )
