import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

diMuonHistos = cms.EDAnalyzer("DiMuonHistograms",
                              MuonCollection = cms.InputTag("muons"),
                              VertexLabel     = cms.InputTag("offlinePrimaryVertices"),
                              BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),                              

                              etaBin = cms.int32(400),
                              etaBBin = cms.int32(400),
                              etaEBin = cms.int32(200),
                            
                              etaBinLM = cms.int32(0),
                              etaBBinLM = cms.int32(0),
                              etaEBinLM = cms.int32(0),
                             
                              etaBMin = cms.double(0.),
                              etaBMax = cms.double(1.1),
                              etaECMin = cms.double(0.9),
                              etaECMax = cms.double(2.4),
                              
                              LowMassMin = cms.double(2.0),
                              LowMassMax = cms.double(55.0),
                              HighMassMin = cms.double(55.0),
                              HighMassMax = cms.double(155.0)
                              )
