import FWCore.ParameterSet.Config as cms

hltFatJetMassFilter = cms.EDFilter("HLTFatJetMassFilter",
    inputJetTag = cms.InputTag( "hltAntiKT5L2L3CorrCaloJets"),
    saveTags = cms.bool( False ),
    minMass = cms.double(100.),
    fatJetDeltaR = cms.double(1.1),
    maxDeltaEta = cms.double(2.0),
)


