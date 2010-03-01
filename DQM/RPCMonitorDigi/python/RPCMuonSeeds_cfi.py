import FWCore.ParameterSet.Config as cms

#STANDALONE MUON RECO
rpcMuonSeed = cms.EDProducer("RPCSeedGenerator",
    RPCRecHitsLabel = cms.InputTag("rpcRecHits")
)


