import FWCore.ParameterSet.Config as cms

slimmedElectrons = cms.EDProducer(
    "ModifiedElectronProducer",
    src = cms.InputTag("slimmedElectrons"),
    modifierConfig = cms.PSet()
)
