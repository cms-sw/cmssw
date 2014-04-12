import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
mcMuonSeeds = cms.EDProducer("MCMuonSeedGenerator2",
                             MuonServiceProxy,
                             CSCSimHit = cms.InputTag("MuonCSCHits","g4SimHits"),
                             ErrorScale = cms.double(10000.0),
                             DTSimHit = cms.InputTag("MuonDTHits","g4SimHits"),
                             RPCSimHit = cms.InputTag("MuonRPCHits","g4SimHits"),
                             SimTrack = cms.InputTag("g4SimHits"),
                             SimVertex = cms.InputTag("g4SimHits"),
                             SeedType = cms.string("FromTracks")
                             )
