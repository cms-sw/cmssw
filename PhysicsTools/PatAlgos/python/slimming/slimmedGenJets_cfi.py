import FWCore.ParameterSet.Config as cms

slimmedGenJets = cms.EDProducer("PATGenJetSlimmer",
    src = cms.InputTag("ak4GenJetsNoNu"),
    packedGenParticles = cms.InputTag("packedGenParticles"),
    cut = cms.string("pt > 8"),
    cutLoose = cms.string(""),
    nLoose = cms.uint32(0),
    clearDaughters = cms.bool(False), #False means rekeying
    dropSpecific = cms.bool(False),
)

slimmedGenJetsAK8 = cms.EDProducer("PATGenJetSlimmer",
    src = cms.InputTag("ak8GenJetsNoNu"),
    packedGenParticles = cms.InputTag("packedGenParticles"),
    cut = cms.string("pt > 80"),
    cutLoose = cms.string("pt > 10."),
    nLoose = cms.uint32(3),
    clearDaughters = cms.bool(False), #False means rekeying
    dropSpecific = cms.bool(False),
)
