import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
muonRecoAnalyzer = DQMEDAnalyzer('MuonRecoAnalyzer',
                                  MuonServiceProxy, 
                                  MuonCollection       = cms.InputTag("muons"),
                                  inputTagVertex       = cms.InputTag("offlinePrimaryVertices"),
                                  inputTagBeamSpot     = cms.InputTag("offlineBeamSpot"),
                                  IsminiAOD            = cms.bool( False ),
                                  doMVA                = cms.bool( False ),
                                  useGEM               = cms.untracked.bool(False),
                                  maxGEMhitsSoftMuonMVA = cms.untracked.int32(6),
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
                                  tunePMax = cms.double(1.0),
                                  folder = cms.string("Muons/MuonRecoAnalyzer")
                                  )
muonRecoAnalyzer_miniAOD = DQMEDAnalyzer('MuonRecoAnalyzer',
                                          MuonServiceProxy, 
                                          MuonCollection = cms.InputTag("slimmedMuons"),
                                          inputTagVertex       = cms.InputTag("offlinePrimaryVertices"),
                                          inputTagBeamSpot     = cms.InputTag("offlineBeamSpot"),
                                          IsminiAOD            = cms.bool( True ),
                                          doMVA                = cms.bool( False ),
                                          useGEM               = cms.untracked.bool(False),
                                          maxGEMhitsSoftMuonMVA = cms.untracked.int32(6),
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
                                          tunePMax = cms.double(1.0),
                                          folder = cms.string("Muons_miniAOD/MuonRecoAnalyzer")
                                          )

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify(muonRecoAnalyzer, useGEM=cms.untracked.bool(True))
run3_GEM.toModify(muonRecoAnalyzer_miniAOD, useGEM=cms.untracked.bool(True))

from Configuration.Eras.Modifier_phase2_GEM_cff import phase2_GEM
phase2_GEM.toModify(muonRecoAnalyzer, maxGEMhitsSoftMuonMVA=cms.untracked.int32(22))
phase2_GEM.toModify(muonRecoAnalyzer_miniAOD, maxGEMhitsSoftMuonMVA=cms.untracked.int32(22))
