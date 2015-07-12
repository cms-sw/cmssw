import FWCore.ParameterSet.Config as cms

modifiedJets = cms.EDProducer(
    "ModifiedJetProducer",
    src = cms.InputTag("slimmedJets",processName=cms.InputTag.skipCurrentProcess()),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)
