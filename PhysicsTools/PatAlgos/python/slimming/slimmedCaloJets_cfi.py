import FWCore.ParameterSet.Config as cms

slimmedCaloJets = cms.EDProducer("CaloJetSlimmer",
    src = cms.InputTag("ak4CaloJets"),
    cut = cms.string("pt>20")
)

