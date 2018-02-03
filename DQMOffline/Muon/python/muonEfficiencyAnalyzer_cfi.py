import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
TightMuonEfficiencyAnalyzer = DQMEDAnalyzer('EfficiencyAnalyzer',
                                             MuonServiceProxy,
                                             MuonCollection  = cms.InputTag("muons"),
                                             TrackCollection = cms.InputTag("generalTracks"),
                                             VertexLabel     = cms.InputTag("offlinePrimaryVertices"),
                                             BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),
                                             
                                             doPrimaryVertexCheck = cms.bool( True ),
                                             
                                             ptBin = cms.int32(10),
                                             ptMax = cms.double(100),
                                             ptMin = cms.double(10),
                                             
                                             etaBin = cms.int32(8),
                                             etaMax = cms.double(2.5),
                                             etaMin = cms.double(-2.5),
                                             
                                             phiBin = cms.int32(8),
                                             phiMax = cms.double(3.2),
                                             phiMin = cms.double(-3.2),
                                             
                                             vtxBin = cms.int32(10),
                                             vtxMax = cms.double(40.5),
                                             vtxMin = cms.double(0.5),
                                             
                                             ID = cms.string("Tight"),
                                             folder = cms.string("Muons/EfficiencyAnalyzer/")
                                             )
TightMuonEfficiencyAnalyzer_miniAOD = DQMEDAnalyzer('EfficiencyAnalyzer',
                                                     MuonServiceProxy,
                                                     MuonCollection  = cms.InputTag("slimmedMuons"),
                                                     TrackCollection = cms.InputTag("generalTracks"),
                                                     VertexLabel     = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                                     BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),
                                                     
                                                     doPrimaryVertexCheck = cms.bool( True ),
                                                     
                                                     ptBin = cms.int32(10),
                                                     ptMax = cms.double(100),
                                                     ptMin = cms.double(10),
                                        
                                                     etaBin = cms.int32(8),
                                                     etaMax = cms.double(2.5),
                                                     etaMin = cms.double(-2.5),
                                                     
                                                     phiBin = cms.int32(8),
                                                     phiMax = cms.double(3.2),
                                                     phiMin = cms.double(-3.2),
                                                     
                                                     vtxBin = cms.int32(10),
                                                     vtxMax = cms.double(40.5),
                                                     vtxMin = cms.double(0.5),
                                                     
                                                     ID = cms.string("Tight"),
                                                     folder = cms.string("Muons_miniAOD/EfficiencyAnalyzer/")
                                                     )


LooseMuonEfficiencyAnalyzer = DQMEDAnalyzer('EfficiencyAnalyzer',
                                             MuonServiceProxy,
                                             MuonCollection  = cms.InputTag("muons"),
                                             TrackCollection = cms.InputTag("generalTracks"),
                                             VertexLabel     = cms.InputTag("offlinePrimaryVertices"),
                                             BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),
                                             
                                             doPrimaryVertexCheck = cms.bool( True ),
                                             
                                             ptBin = cms.int32(10),
                                             ptMax = cms.double(100),
                                             ptMin = cms.double(10),
                                             
                                             etaBin = cms.int32(8),
                                             etaMax = cms.double(2.5),
                                             etaMin = cms.double(-2.5),
                                             
                                             phiBin = cms.int32(8),
                                             phiMax = cms.double(3.2),
                                             phiMin = cms.double(-3.2),
                                             
                                             vtxBin = cms.int32(10),
                                             vtxMax = cms.double(40.5),
                                             vtxMin = cms.double(0.5),
                                             
                                             ID = cms.string("Loose"),
                                             folder = cms.string("Muons/EfficiencyAnalyzer/")
                                             
                                             )
LooseMuonEfficiencyAnalyzer_miniAOD = DQMEDAnalyzer('EfficiencyAnalyzer',
                                                     MuonServiceProxy,
                                                     MuonCollection  = cms.InputTag("slimmedMuons"),
                                                     TrackCollection = cms.InputTag("generalTracks"),
                                                     VertexLabel     = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                                     BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),
                                                     
                                                     doPrimaryVertexCheck = cms.bool( True ),
                                                     
                                                     ptBin = cms.int32(10),
                                                     ptMax = cms.double(100),
                                                     ptMin = cms.double(10),
                                                     
                                                     etaBin = cms.int32(8),
                                                     etaMax = cms.double(2.5),
                                                     etaMin = cms.double(-2.5),
                                                     
                                                     phiBin = cms.int32(8),
                                                     phiMax = cms.double(3.2),
                                                     phiMin = cms.double(-3.2),
                                                     
                                                     vtxBin = cms.int32(10),
                                                     vtxMax = cms.double(40.5),
                                                     vtxMin = cms.double(0.5),

                                                     ID = cms.string("Loose"),
                                                     folder = cms.string("Muons_miniAOD/EfficiencyAnalyzer/")
                                                     
                                                     )


MediumMuonEfficiencyAnalyzer = DQMEDAnalyzer('EfficiencyAnalyzer',
                                              MuonServiceProxy,
                                              MuonCollection  = cms.InputTag("muons"),
                                              TrackCollection = cms.InputTag("generalTracks"),
                                              VertexLabel     = cms.InputTag("offlinePrimaryVertices"),
                                              BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),
                                              
                                              doPrimaryVertexCheck = cms.bool( True ),
                                              
                                              ptBin = cms.int32(10),
                                              ptMax = cms.double(100),
                                              ptMin = cms.double(10),
                                              
                                              etaBin = cms.int32(8),
                                              etaMax = cms.double(2.5),
                                              etaMin = cms.double(-2.5),
                                              
                                              phiBin = cms.int32(8),
                                              phiMax = cms.double(3.2),
                                              phiMin = cms.double(-3.2),
                                              
                                              vtxBin = cms.int32(10),
                                              vtxMax = cms.double(40.5),
                                              vtxMin = cms.double(0.5),
                                              
                                              ID = cms.string("Medium"),
                                              folder = cms.string("Muons/EfficiencyAnalyzer/")
                                              )
MediumMuonEfficiencyAnalyzer_miniAOD = DQMEDAnalyzer('EfficiencyAnalyzer',
                                                      MuonServiceProxy,
                                                      MuonCollection  = cms.InputTag("slimmedMuons"),
                                                      TrackCollection = cms.InputTag("generalTracks"),
                                                      VertexLabel     = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                                      BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),
                                                      
                                                      doPrimaryVertexCheck = cms.bool( True ),
                                                      
                                                      ptBin = cms.int32(10),
                                                      ptMax = cms.double(100),
                                                      ptMin = cms.double(10),
                                                      
                                                      etaBin = cms.int32(8),
                                                      etaMax = cms.double(2.5),
                                                      etaMin = cms.double(-2.5),
                                              
                                                      phiBin = cms.int32(8),
                                                      phiMax = cms.double(3.2),
                                                      phiMin = cms.double(-3.2),
                                                      
                                                      vtxBin = cms.int32(10),
                                                      vtxMax = cms.double(40.5),
                                                      vtxMin = cms.double(0.5),
                                                      
                                                      ID = cms.string("Medium"),
                                                      folder = cms.string("Muons_miniAOD/EfficiencyAnalyzer/")
                                                      
                                                      )
