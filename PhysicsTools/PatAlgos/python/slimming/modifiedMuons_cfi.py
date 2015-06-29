import FWCore.ParameterSet.Config as cms

slimmedMuons = cms.EDProducer(
    "ModifiedMuonProducer",
    src = cms.InputTag("slimmedMuons::PAT"),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)
