import FWCore.ParameterSet.Config as cms

hltL1SeedForDoublePuppiTau = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pDoublePuppiTau52_52')
)
