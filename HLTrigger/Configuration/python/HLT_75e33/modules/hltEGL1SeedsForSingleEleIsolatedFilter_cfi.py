import FWCore.ParameterSet.Config as cms

hltEGL1SeedsForSingleEleIsolatedFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pSingleEGEle51 or pSingleTkEle36 or pSingleIsoTkEle28')
)
