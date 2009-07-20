import FWCore.ParameterSet.Config as cms

potentialTOBFrameHeaderEventsSPeak = cms.EDFilter('EventWithHistoryEDFilter',
                                                  partitionName = cms.untracked.string("TO"),
                                                  absBXInCycleRangeLtcyAware = cms.untracked.vint32(38,40)
                                                 )
