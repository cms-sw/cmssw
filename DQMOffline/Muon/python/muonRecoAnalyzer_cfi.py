import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

muonRecoAnalyzer = cms.EDAnalyzer("MuonRecoAnalyzer",
                                  MuonServiceProxy, 
                                  MuonCollection       = cms.InputTag("muons"),
                                  
                                  # histograms parameters
                                  thetaBin = cms.int32(100),
                                  thetaMin = cms.double(0.0),
                                  thetaMax = cms.double(3.2),
                                  
                                  pResBin = cms.int32(50),
                                  pResMin = cms.double(-0.01),
                                  pResMax = cms.double(0.01),

                                  rhBin = cms.int32(25),
                                  rhMin = cms.double(0.0),
                                  rhMax = cms.double(1.001),
                                  
                                  etaBin = cms.int32(100),
                                  etaMin = cms.double(-3.0),
                                  etaMax = cms.double(3.0),                                  
                                  
                                  chi2Bin = cms.int32(100),
                                  chi2Min = cms.double(0),
                                  chi2Max = cms.double(20),
                                  
                                  pBin = cms.int32(500),
                                  pMin = cms.double(0.0),
                                  pMax = cms.double(500.0),
                                  
                                  phiBin = cms.int32(100),
                                  phiMin = cms.double(-3.2),
                                  phiMax = cms.double(3.2),
                                  
                                  ptBin = cms.int32(500),
                                  ptMin = cms.double(0.0),
                                  ptMax = cms.double(500.0),
                                  
                                  tunePBin = cms.int32(100),
                                  tunePMin = cms.double(-1.0),
                                  tunePMax = cms.double(1.0)
                                  )
