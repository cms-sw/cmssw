import FWCore.ParameterSet.Config as cms

# producer for alcahbhemuon (HCAL with muons)
HBHEMuonProd = cms.EDProducer("AlCaHBHEMuonProducer",
                              BeamSpotLabel     = cms.InputTag("offlineBeamSpot"),
                              VertexLabel       = cms.InputTag("offlinePrimaryVertices"),
                              EBRecHitLabel     = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                              EERecHitLabel     = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                              HBHERecHitLabel   = cms.InputTag("hbhereco"),
                              MuonLabel         = cms.InputTag("muons"),
                              MinimumMuonP      = cms.double(10.0),
                              )

