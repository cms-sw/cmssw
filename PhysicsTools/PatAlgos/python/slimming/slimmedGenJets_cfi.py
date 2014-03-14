import FWCore.ParameterSet.Config as cms

slimmedGenJets = cms.EDProducer("PATGenJetSlimmer",
    src = cms.InputTag("ak5GenJets"),
    cut = cms.string("pt > 8"),
    clearDaughters = cms.bool(True),
    dropSpecific = cms.bool(False),
)
