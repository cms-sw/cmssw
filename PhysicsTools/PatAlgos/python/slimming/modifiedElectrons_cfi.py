import FWCore.ParameterSet.Config as cms

slimmedElectrons = cms.EDProducer(
    "ModifiedElectronProducer",
    src = cms.InputTag("slimmedElectrons::PAT"),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)
