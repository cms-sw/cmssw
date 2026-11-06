import FWCore.ParameterSet.Config as cms

simBmtfDigis = cms.EDProducer("L1TMuonBarrelTrackProducer",
    DTDigi_Source = cms.InputTag("simTwinMuxDigis"),
    DTDigi_Theta_Source = cms.InputTag("simTwinMuxDigis"),
    Debug = cms.untracked.int32(0)
)
