import FWCore.ParameterSet.Config as cms

hltRPCFilter= cms.EDFilter("HLTRPCFilter",
  rangestrips = cms.untracked.double(4.),
# rpcRecHits = cms.InputTag('rpcRecHits'),
  rpcRecHits = cms.InputTag('hltRpcRecHits'),
  rpcDTPoints = cms.InputTag("rpcPointProducer","RPCDTExtrapolatedPoints"),
  rpcCSCPoints = cms.InputTag("rpcPointProducer","RPCCSCExtrapolatedPoints")
)
