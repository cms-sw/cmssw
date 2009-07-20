import FWCore.ParameterSet.Config as cms

latencyPlusOne = cms.EDFilter('EventWithHistoryEDFilter',
                             dbxRangeLtcyAware = cms.untracked.vint32(1,1),
                             )
