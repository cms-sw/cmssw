import FWCore.ParameterSet.Config as cms

slimmedJets = cms.EDProducer(
    "ModifiedJetProducer",
    src = cms.InputTag("slimmedJets::PAT"),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

slimmedJetsAK8 = cms.EDProducer(
    "ModifiedJetProducer",
    src = cms.InputTag("slimmedJetsAK8::PAT"),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

slimmedJetsPuppi = cms.EDProducer(
    "ModifiedJetProducer",
    src = cms.InputTag("slimmedJetsPuppi::PAT"),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)
