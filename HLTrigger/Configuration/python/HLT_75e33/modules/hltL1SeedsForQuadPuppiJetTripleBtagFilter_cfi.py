import FWCore.ParameterSet.Config as cms

hltL1SeedsForQuadPuppiJetTripleBtagFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pPuppiHT400 and pQuadJet70_55_40_40')
)
