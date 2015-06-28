import FWCore.ParameterSet.Config as cms

slimmedJets = cms.EDProducer(
    "ModifiedJetProducer",
    src = cms.InputTag("slimmedJets"),
    modifierConfig = cms.PSet()
)

slimmedJetsAK8 = cms.EDProducer(
    "ModifiedJetProducer",
    src = cms.InputTag("slimmedJetsAK8"),
    modifierConfig = cms.PSet()
)
