import FWCore.ParameterSet.Config as cms

hltEGL1SeedsForDoubleEleNonIsolatedFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('L1T_TkEm37TkEm24 or L1T_TkEle25TkEle12')
)
