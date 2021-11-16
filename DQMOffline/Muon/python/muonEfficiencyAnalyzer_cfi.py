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
                                             
                                             vtxBin = cms.int32(30),
                                             vtxMax = cms.double(149.5),
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
                                                     
                                                     vtxBin = cms.int32(30),
                                                     vtxMax = cms.double(149.5),
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
                                             
                                             vtxBin = cms.int32(30),
                                             vtxMax = cms.double(149.5),
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
                                                     
                                                     vtxBin = cms.int32(30),
                                                     vtxMax = cms.double(149.5),
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
                                              
                                              vtxBin = cms.int32(30),
                                              vtxMax = cms.double(149.5),
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
                                                      
                                                      vtxBin = cms.int32(30),
                                                      vtxMax = cms.double(149.5),
                                                      vtxMin = cms.double(0.5),                                                      
                                                      ID = cms.string("Medium"),
                                                      folder = cms.string("Muons_miniAOD/EfficiencyAnalyzer/")
                                                      )


EfficiencyAnalyzer = cms.Sequence(TightMuonEfficiencyAnalyzer*LooseMuonEfficiencyAnalyzer*MediumMuonEfficiencyAnalyzer)

EfficiencyAnalyzer_miniAOD = cms.Sequence(TightMuonEfficiencyAnalyzer_miniAOD*LooseMuonEfficiencyAnalyzer_miniAOD*MediumMuonEfficiencyAnalyzer_miniAOD)

TightMuonEfficiencyAnalyzer_Phase2=TightMuonEfficiencyAnalyzer.clone()
TightMuonEfficiencyAnalyzer_Phase2.vtxBin=20
TightMuonEfficiencyAnalyzer_Phase2.vtxMax=249.5
TightMuonEfficiencyAnalyzer_Phase2.vtxMin=149.5

LooseMuonEfficiencyAnalyzer_Phase2=LooseMuonEfficiencyAnalyzer.clone()                                                                      
LooseMuonEfficiencyAnalyzer_Phase2.vtxBin=20                                                                                                
LooseMuonEfficiencyAnalyzer_Phase2.vtxMin=149.5  
LooseMuonEfficiencyAnalyzer_Phase2.vtxMax=249.5

MediumMuonEfficiencyAnalyzer_Phase2=MediumMuonEfficiencyAnalyzer.clone()                                                                    
MediumMuonEfficiencyAnalyzer_Phase2.vtxBin=20                                                                                               
MediumMuonEfficiencyAnalyzer_Phase2.vtxMin=149.5  
MediumMuonEfficiencyAnalyzer_Phase2.vtxMax=249.5 


TightMuonEfficiencyAnalyzer_miniAOD_Phase2=TightMuonEfficiencyAnalyzer_miniAOD.clone()                                                      
TightMuonEfficiencyAnalyzer_miniAOD_Phase2.vtxBin=20                                                                                       
TightMuonEfficiencyAnalyzer_miniAOD_Phase2.vtxMax=249.5                                                                                     
TightMuonEfficiencyAnalyzer_miniAOD_Phase2.vtxMin=149.5                                                                                    
     

LooseMuonEfficiencyAnalyzer_miniAOD_Phase2=LooseMuonEfficiencyAnalyzer_miniAOD.clone()                                                      
LooseMuonEfficiencyAnalyzer_miniAOD_Phase2.vtxBin=20                                                                                       
LooseMuonEfficiencyAnalyzer_miniAOD_Phase2.vtxMin=149.5                                                                                     
LooseMuonEfficiencyAnalyzer_miniAOD_Phase2.vtxMax=249.5                                                                                    

MediumMuonEfficiencyAnalyzer_miniAOD_Phase2=MediumMuonEfficiencyAnalyzer.clone()                                                           
MediumMuonEfficiencyAnalyzer_miniAOD_Phase2.vtxBin=20                                                                                       
MediumMuonEfficiencyAnalyzer_miniAOD_Phase2.vtxMin=149.5                                                                                    
MediumMuonEfficiencyAnalyzer_miniAOD_Phase2.vtxMax=249.5                                                                                   
                                                        

EfficiencyAnalyzer_Phase2 = cms.Sequence(TightMuonEfficiencyAnalyzer_Phase2*LooseMuonEfficiencyAnalyzer_Phase2*MediumMuonEfficiencyAnalyzer_Phase2)

EfficiencyAnalyzer_miniAOD_Phase2 = cms.Sequence(TightMuonEfficiencyAnalyzer_miniAOD_Phase2*LooseMuonEfficiencyAnalyzer_miniAOD_Phase2*MediumMuonEfficiencyAnalyzer_miniAOD_Phase2)                                                                                                    

            
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon                                                                         
phase2_muon.toReplaceWith(EfficiencyAnalyzer, EfficiencyAnalyzer_Phase2)                                                                   
phase2_muon.toReplaceWith(EfficiencyAnalyzer_miniAOD, EfficiencyAnalyzer_miniAOD_Phase2)    
                                                                                    




