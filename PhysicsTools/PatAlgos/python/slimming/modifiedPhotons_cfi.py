import FWCore.ParameterSet.Config as cms

slimmedPhotons = cms.EDProducer(
    "ModifiedPhotonProducer",
    src = cms.InputTag("slimmedPhotons"),
    modifierConfig = cms.PSet()
)
