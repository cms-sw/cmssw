import FWCore.ParameterSet.Config as cms

#STANDALONE MUON RECO
rpcMuonSeed = cms.EDFilter("RPCSeedGenerator",
    RPCRecHitsLabel = cms.InputTag("rpcRecHits")
)


