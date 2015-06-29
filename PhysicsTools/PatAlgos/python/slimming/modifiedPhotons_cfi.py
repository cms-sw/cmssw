import FWCore.ParameterSet.Config as cms

slimmedPhotons = cms.EDProducer(
    "ModifiedPhotonProducer",
    src = cms.InputTag("slimmedPhotons::PAT"),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)
