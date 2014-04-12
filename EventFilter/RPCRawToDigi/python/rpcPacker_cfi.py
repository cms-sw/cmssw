import FWCore.ParameterSet.Config as cms

rpcpacker = cms.EDProducer("RPCPackingModule",
  InputLabel = cms.InputTag("simMuonRPCDigis")
)


