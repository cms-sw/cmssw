import FWCore.ParameterSet.Config as cms

rpcRecHitAli = cms.EDProducer("RPCRecHitAli",
  debug = cms.untracked.bool(True),
  rpcRecHits = cms.InputTag("rpcRecHits"),
  AliFileName = cms.untracked.string('/afs/cern.ch/user/c/carrillo/efficiency/CMSSW_3_8_1/src/RecoLocalMuon/RPCRecHit/data/Alignment.dat')
)
