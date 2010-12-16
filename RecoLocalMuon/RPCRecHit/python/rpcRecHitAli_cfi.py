import FWCore.ParameterSet.Config as cms

rpcRecHitAli = cms.EDProducer("RPCRecHitAli",
  debug = cms.untracked.bool(True),
  rpcRecHits = cms.InputTag("rpcRecHits"),
  AliFileName = cms.untracked.string('../data/Alignment.dat')
)
