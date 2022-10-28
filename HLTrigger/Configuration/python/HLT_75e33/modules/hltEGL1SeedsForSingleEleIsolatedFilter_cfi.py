import FWCore.ParameterSet.Config as cms

hltEGL1SeedsForSingleEleIsolatedFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('L1T_TkEm51 or L1T_TkEle36 or L1T_TkIsoEle28')
)
