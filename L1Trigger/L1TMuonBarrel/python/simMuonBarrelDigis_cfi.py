import FWCore.ParameterSet.Config as cms

simTwinMuxDigis = cms.EDProducer('L1TTwinMuxProducer', 
    DTDigi_Source = cms.InputTag("simDtTriggerPrimitiveDigis"),
    DTThetaDigi_Source = cms.InputTag("simDtTriggerPrimitiveDigis"),
    RPC_Source = cms.InputTag("simMuonRPCDigis"),
)

simBmtfDigis = cms.EDProducer("L1TMuonBarrelTrackProducer",
    Debug = cms.untracked.int32(0),
    DTDigi_Source = cms.InputTag("simTwinMuxDigis"),
    DTDigi_Theta_Source = cms.InputTag("simDtTriggerPrimitiveDigis"),
)


