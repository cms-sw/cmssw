import FWCore.ParameterSet.Config as cms

modifiedMuons = cms.EDProducer(
    "ModifiedMuonProducer",
    src = cms.InputTag("slimmedMuons",processName=cms.InputTag.skipCurrentProcess()),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)
