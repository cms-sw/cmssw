import FWCore.ParameterSet.Config as cms

slimmedJets = cms.EDProducer(
    "ModifiedJetProducer",
    src = cms.InputTag("slimmedJets",processName=cms.InputTag.skipCurrentProcess()),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

slimmedJetsAK8 = cms.EDProducer(
    "ModifiedJetProducer",
    src = cms.InputTag("slimmedJetsAK8",processName=cms.InputTag.skipCurrentProcess()),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

slimmedJetsPuppi = cms.EDProducer(
    "ModifiedJetProducer",
    src = cms.InputTag("slimmedJetsPuppi",processName=cms.InputTag.skipCurrentProcess()),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)
