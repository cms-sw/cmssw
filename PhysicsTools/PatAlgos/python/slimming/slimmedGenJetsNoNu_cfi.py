import FWCore.ParameterSet.Config as cms

slimmedGenJetsNoNu = cms.EDProducer("PATGenJetSlimmer",
    src = cms.InputTag("ak4GenJetsNoNu"),
    packedGenParticles = cms.InputTag("packedGenParticles"),
    cut = cms.string("pt > 8"),
    clearDaughters = cms.bool(False), #False means rekeying
    dropSpecific = cms.bool(False),
)
