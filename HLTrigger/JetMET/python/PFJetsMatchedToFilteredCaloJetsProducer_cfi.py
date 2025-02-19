import FWCore.ParameterSet.Config as cms

PFJetsMatchedToFilteredCaloJetsProducer= cms.EDProducer( "PFJetsMatchedToFilteredCaloJetsProducer",
    PFJetSrc = cms.InputTag( 'hltAntiKT5PFJets' ),
    CaloJetFilter = cms.InputTag( "hltSingleJet240Regional" ),
    DeltaR = cms.double(0.5)
)

