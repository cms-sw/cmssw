import FWCore.ParameterSet.Config as cms

potentialTIBTECFrameHeaderEventsWide = cms.EDFilter('EventWithHistoryEDFilter',
                                                 partitionName = cms.untracked.string("TI"),
                                                 absBXInCycleRangeLtcyAware = cms.untracked.vint32(16,44)
                                                 )
