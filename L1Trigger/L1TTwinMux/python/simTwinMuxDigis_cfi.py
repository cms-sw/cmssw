import FWCore.ParameterSet.Config as cms

simTwinMuxDigisEmu = cms.EDProducer('L1TwinMuxProducer', 
    DTDigi_Source = cms.InputTag("simDtTriggerPrimitiveDigis"),
    DTThetaDigi_Source = cms.InputTag("simDtTriggerPrimitiveDigis"),
    RPC_Source = cms.InputTag("simMuonRPCDigis"),
)
