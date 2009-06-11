import FWCore.ParameterSet.Config as cms

rpcpacker = cms.EDFilter("RPCPackingModule",
  InputLabel = cms.InputTag("simMuonRPCDigis")
)


