import FWCore.ParameterSet.Config as cms

hltRPCFilter= cms.EDFilter("HLTRPCFilter",
  rangestrips = cms.untracked.double(4.),
  rpcRecHits = cms.InputTag('rpcRecHits'),
  rpcDTPoints = cms.InputTag("rpcPointProducer","RPCDTExtrapolatedPoints"),
  rpcCSCPoints = cms.InputTag("rpcPointProducer","RPCCSCExtrapolatedPoints")
)
