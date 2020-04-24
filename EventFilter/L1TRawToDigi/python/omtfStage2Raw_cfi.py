import FWCore.ParameterSet.Config as cms

omtfStage2Raw = cms.EDProducer("OmtfPacker",
  rpcInputLabel = cms.InputTag('simMuonRPCDigis'),
  cscInputLabel = cms.InputTag('simCscTriggerPrimitiveDigis'),
  dtPhInputLabel = cms.InputTag('simTwinMuxDigis'),
  dtThInputLabel = cms.InputTag('simTwinMuxDigis'),
)
