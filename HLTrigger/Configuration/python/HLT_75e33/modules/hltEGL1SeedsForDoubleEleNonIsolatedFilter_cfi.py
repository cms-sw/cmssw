import FWCore.ParameterSet.Config as cms

hltEGL1SeedsForDoubleEleNonIsolatedFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pDoubleEGEle37_24 or pDoubleTkEle25_12')
)
