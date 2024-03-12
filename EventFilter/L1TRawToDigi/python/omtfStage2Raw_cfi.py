import FWCore.ParameterSet.Config as cms

omtfStage2Raw = cms.EDProducer("OmtfPacker",
  rpcInputLabel = cms.InputTag('simMuonRPCDigis'),
  cscInputLabel = cms.InputTag('simCscTriggerPrimitiveDigis'),
  dtPhInputLabel = cms.InputTag('simTwinMuxDigis'),
  dtThInputLabel = cms.InputTag('simTwinMuxDigis'),
)
# foo bar baz
# FVHznH0S3lDaz
# Bd6WxX4JCdNd7
