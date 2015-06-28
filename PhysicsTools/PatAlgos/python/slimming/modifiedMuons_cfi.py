import FWCore.ParameterSet.Config as cms

slimmedMuons = cms.EDProducer(
    "ModifiedMuonProducer",
    src = cms.InputTag("slimmedMuons"),
    modifierConfig = cms.PSet()
)
