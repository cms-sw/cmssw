import FWCore.ParameterSet.Config as cms

hltEGL1SeedsForSingleEleNonIsolatedFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('L1T_TkEm51 or L1T_TkEle36')
)
