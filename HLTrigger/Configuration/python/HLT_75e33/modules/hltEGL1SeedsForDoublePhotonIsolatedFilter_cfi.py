import FWCore.ParameterSet.Config as cms

hltEGL1SeedsForDoublePhotonIsolatedFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pDoubleEGEle37_24 or pDoubleIsoTkPho22_12')
)
