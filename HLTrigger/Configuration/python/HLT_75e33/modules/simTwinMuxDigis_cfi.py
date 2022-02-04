import FWCore.ParameterSet.Config as cms

simTwinMuxDigis = cms.EDProducer("L1TTwinMuxProducer",
    DTDigi_Source = cms.InputTag("simDtTriggerPrimitiveDigis"),
    DTThetaDigi_Source = cms.InputTag("simDtTriggerPrimitiveDigis"),
    RPC_Source = cms.InputTag("simMuonRPCDigis")
)
