import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

muonKinVsEtaAnalyzer = cms.EDAnalyzer("MuonKinVsEtaAnalyzer",
                                      MuonServiceProxy,
                                      
                                      MuonCollection  = cms.InputTag("muons"),
                                      VertexLabel     = cms.InputTag("offlinePrimaryVertices"),
                                      BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),
                                   
                                      pBin = cms.int32(100),
                                      pMin = cms.double(0.0),
                                      pMax = cms.double(100.0),
                                      
                                      ptBin = cms.int32(100),
                                      ptMin = cms.double(0.0),
                                      ptMax = cms.double(100.0),
                                      
                                      etaBin = cms.int32(100),
                                      etaMin = cms.double(-3.0),
                                      etaMax = cms.double(3.0),
                                      
                                      phiBin = cms.int32(100),
                                      phiMin = cms.double(-3.2),
                                      phiMax = cms.double(3.2),
                                      
                                      chiBin = cms.int32(100),
                                      chiMin = cms.double(0.),
                                      chiMax = cms.double(20.),
                                      
                                      chiprobMin = cms.double(0.),
                                      chiprobMax = cms.double(1.),
                                      
                                      etaBMin = cms.double(0.),
                                      etaBMax = cms.double(1.1),
                                      etaECMin = cms.double(0.9),
                                      etaECMax = cms.double(2.4),
                                      etaOvlpMin = cms.double(0.9),
                                      etaOvlpMax = cms.double(1.1)
                                      )
