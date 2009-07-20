import FWCore.ParameterSet.Config as cms

tibtecFrameHeaderEvents = cms.EDFilter('EventWithHistoryEDFilter',
                                       partitionName = cms.untracked.string("TI"),
                                       dbxInCycleRangeLtcyAware = cms.untracked.vint32(298,319)
                                       )
