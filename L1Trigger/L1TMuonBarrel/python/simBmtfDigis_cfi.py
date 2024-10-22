import FWCore.ParameterSet.Config as cms

simBmtfDigis = cms.EDProducer("L1TMuonBarrelTrackProducer",
    Debug = cms.untracked.int32(0),
    DTDigi_Source = cms.InputTag("simTwinMuxDigis"),
    DTDigi_Theta_Source = cms.InputTag("simTwinMuxDigis"),
)
