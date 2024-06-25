import FWCore.ParameterSet.Config as cms

hltL1SeedsForDoublePuppiJetBtagFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pDoublePuppiJet112_112')
)
