import FWCore.ParameterSet.Config as cms

potentialTOBFrameHeaderEventsFPeak = cms.EDFilter('EventWithHistoryEDFilter',
                                                  partitionName = cms.untracked.string("TO"),
                                                  absBXInCycleRangeLtcyAware = cms.untracked.vint32(19,22)
                                                 )
