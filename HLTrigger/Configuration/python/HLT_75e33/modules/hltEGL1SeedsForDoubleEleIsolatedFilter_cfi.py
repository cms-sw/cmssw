import FWCore.ParameterSet.Config as cms

hltEGL1SeedsForDoubleEleIsolatedFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pDoubleEGEle37_24 or pDoubleTkEle25_12 or pIsoTkEleEGEle22_12')
)
