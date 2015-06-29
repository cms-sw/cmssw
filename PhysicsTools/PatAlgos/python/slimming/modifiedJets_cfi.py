import FWCore.ParameterSet.Config as cms

slimmedJets = cms.EDProducer(
    "ModifiedJetProducer",
    src = cms.InputTag("slimmedJets"),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

slimmedJetsAK8 = cms.EDProducer(
    "ModifiedJetProducer",
    src = cms.InputTag("slimmedJetsAK8"),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

slimmedJetsPuppi = cms.EDProducer(
    "ModifiedJetProducer",
    src = cms.InputTag("slimmedJetsPuppi"),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)
