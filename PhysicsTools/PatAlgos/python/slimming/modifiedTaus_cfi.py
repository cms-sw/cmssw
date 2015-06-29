import FWCore.ParameterSet.Config as cms

slimmedTaus = cms.EDProducer(
    "ModifiedTauProducer",
    src = cms.InputTag("slimmedTaus"),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)
