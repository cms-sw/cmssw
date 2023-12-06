import FWCore.ParameterSet.Config as cms

from RecoLocalMuon.RPCRecHit.rpcRecHits_cfi import rpcRecHits
rpcRecHitsForEMTF = rpcRecHits.clone(rpcDigiLabel = 'simMuonRPCDigis')

