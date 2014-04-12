import FWCore.ParameterSet.Config as cms

latencyPlusOne = cms.EDFilter('EventWithHistoryEDFilter',
                              filterConfigurations = cms.untracked.VPSet(
                                 cms.PSet(dbxRangeLtcyAware = cms.untracked.vint32(1,1))
                              )
                             )
