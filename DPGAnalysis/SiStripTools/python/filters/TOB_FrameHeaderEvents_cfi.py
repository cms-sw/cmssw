import FWCore.ParameterSet.Config as cms

tobFrameHeaderEvents = cms.EDFilter('EventWithHistoryEDFilter',
                                    partitionName = cms.untracked.string("TO"),
                                    dbxInCycleRangeLtcyAware = cms.untracked.vint32(298,319)
                                    )
