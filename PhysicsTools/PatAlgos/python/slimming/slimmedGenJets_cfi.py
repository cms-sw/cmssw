import FWCore.ParameterSet.Config as cms

slimmedGenJets = cms.EDProducer("PATGenJetSlimmer",
    src = cms.InputTag("ak4GenJetsNoNu"),
    packedGenParticles = cms.InputTag("packedGenParticles"),
    cut = cms.string("pt > 8"),
    clearDaughters = cms.bool(False), #False means rekeying
    dropSpecific = cms.bool(False),
)

slimmedGenJetsFlavourInfos = cms.EDProducer("GenJetFlavourInfoPreserver",
    GenJets = cms.InputTag("ak4GenJetsNoNu"),
    slimmedGenJets = cms.InputTag("slimmedGenJets"),
    cut = cms.string("pt > 8"), #VERY IMPORTANT: cut needs to be the same as in slimmedGenJets above.                                       
    GenJetFlavourInfos = cms.InputTag("ak4GenJetFlavourInfos")
)



slimmedGenJetsAK8 = cms.EDProducer("PATGenJetSlimmer",
    src = cms.InputTag("ak8GenJetsNoNu"),
    packedGenParticles = cms.InputTag("packedGenParticles"),
    cut = cms.string("pt > 150"),
    clearDaughters = cms.bool(False), #False means rekeying
    dropSpecific = cms.bool(False),
)
