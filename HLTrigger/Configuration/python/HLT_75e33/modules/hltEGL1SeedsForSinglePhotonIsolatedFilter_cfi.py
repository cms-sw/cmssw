import FWCore.ParameterSet.Config as cms

hltEGL1SeedsForSinglePhotonIsolatedFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('L1T_TkEm51 or L1T_TkIsoEm36')
)
