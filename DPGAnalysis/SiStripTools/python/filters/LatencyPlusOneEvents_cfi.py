import FWCore.ParameterSet.Config as cms

latencyPlusOne = cms.EDFilter('EventWithHistoryEDFilter',
                              filterConfigurations = cms.untracked.VPSet(
                                 cms.PSet(dbxRangeLtcyAware = cms.untracked.vint32(1,1))
                              )
                             )
# foo bar baz
# 9Xctjj6Hw2aSY
# BdaZQ6kfgSS5V
