import FWCore.ParameterSet.Config as cms

hltL1SeedsForPuppiHTFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pPuppiHT450')
)
