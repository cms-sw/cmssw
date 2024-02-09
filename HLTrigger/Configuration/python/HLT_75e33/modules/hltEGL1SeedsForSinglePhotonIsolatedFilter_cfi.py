import FWCore.ParameterSet.Config as cms

hltEGL1SeedsForSinglePhotonIsolatedFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pSingleEGEle51 or pSingleIsoTkPho36')
)
