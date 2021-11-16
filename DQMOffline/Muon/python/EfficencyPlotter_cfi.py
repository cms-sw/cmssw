import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

effPlotterLoose = DQMEDHarvester("EfficiencyPlotter",
                                  folder = cms.string("Muons/EfficiencyAnalyzer"),
                                  phiMin = cms.double(-3.2),
                                  etaMin = cms.double(-2.5),
                                  ptMin  = cms.double(10),
                                  etaBin = cms.int32(8),
                                  ptBin = cms.int32(10),
                                  phiBin = cms.int32(8),
                                  etaMax = cms.double(2.5),
                                  phiMax = cms.double(3.2),
                                  ptMax  = cms.double(100),
                                  vtxBin = cms.int32(30),
                                  vtxMin = cms.double(0.5),
                                  vtxMax = cms.double(149.5),
                                  MuonID = cms.string("Loose")
                                  )


effPlotterMedium = DQMEDHarvester("EfficiencyPlotter",
                                   folder = cms.string("Muons/EfficiencyAnalyzer"),
                                   phiMin = cms.double(-3.2),
                                   etaMin = cms.double(-2.5),
                                   ptMin  = cms.double(10),
                                   etaBin = cms.int32(8),
                                   ptBin = cms.int32(10),
                                   phiBin = cms.int32(8),
                                   etaMax = cms.double(2.5),
                                   phiMax = cms.double(3.2),
                                   ptMax  = cms.double(100),
                                   vtxBin = cms.int32(30),
                                   vtxMin = cms.double(0.5),
                                   vtxMax = cms.double(149.5),
                                   MuonID = cms.string("Medium")
                                   )


effPlotterTight = DQMEDHarvester("EfficiencyPlotter",
                                  folder = cms.string("Muons/EfficiencyAnalyzer"),
                                  phiMin = cms.double(-3.2),
                                  etaMin = cms.double(-2.5),
                                  ptMin  = cms.double(10),
                                  etaBin = cms.int32(8),
                                  ptBin = cms.int32(10),
                                  phiBin = cms.int32(8),
                                  etaMax = cms.double(2.5),
                                  phiMax = cms.double(3.2),
                                  ptMax  = cms.double(100),
                                  vtxBin = cms.int32(30),
                                  vtxMin = cms.double(0.5),
                                  vtxMax = cms.double(149.5),
                                  MuonID = cms.string("Tight")
                                  )
effPlotterLooseMiniAOD = DQMEDHarvester("EfficiencyPlotter",
                                          folder = cms.string("Muons_miniAOD/EfficiencyAnalyzer"),
                                          phiMin = cms.double(-3.2),
                                          etaMin = cms.double(-2.5),
                                          ptMin  = cms.double(10),
                                          etaBin = cms.int32(8),
                                          ptBin = cms.int32(10),
                                          phiBin = cms.int32(8),
                                          etaMax = cms.double(2.5),
                                          phiMax = cms.double(3.2),
                                          ptMax  = cms.double(100),
                                          vtxBin = cms.int32(30),
                                          vtxMin = cms.double(0.5),
                                          vtxMax = cms.double(149.5),
                                          MuonID = cms.string("Loose")
                                          )


effPlotterMediumMiniAOD = DQMEDHarvester("EfficiencyPlotter",
                                           folder = cms.string("Muons_miniAOD/EfficiencyAnalyzer"),
                                           phiMin = cms.double(-3.2),
                                           etaMin = cms.double(-2.5),
                                           ptMin  = cms.double(10),
                                           etaBin = cms.int32(8),
                                           ptBin = cms.int32(10),
                                           phiBin = cms.int32(8),
                                           etaMax = cms.double(2.5),
                                           phiMax = cms.double(3.2),
                                           ptMax  = cms.double(100),
                                           vtxBin = cms.int32(30),
                                           vtxMin = cms.double(0.5),
                                           vtxMax = cms.double(149.5),
                                           MuonID = cms.string("Medium")
                                           )


effPlotterTightMiniAOD = DQMEDHarvester("EfficiencyPlotter",
                                          folder = cms.string("Muons_miniAOD/EfficiencyAnalyzer"),
                                          phiMin = cms.double(-3.2),
                                          etaMin = cms.double(-2.5),
                                          ptMin  = cms.double(10),
                                          etaBin = cms.int32(8),
                                          ptBin = cms.int32(10),
                                          phiBin = cms.int32(8),
                                          etaMax = cms.double(2.5),
                                          phiMax = cms.double(3.2),
                                          ptMax  = cms.double(100),
                                          vtxBin = cms.int32(30),
                                          vtxMin = cms.double(0.5),
                                          vtxMax = cms.double(149.5),
                                          MuonID = cms.string("Tight")
                                          )


effPlotter=cms.Sequence(effPlotterLoose*effPlotterMedium*effPlotterTight)
effPlotter_miniAOD=cms.Sequence(effPlotterLooseMiniAOD*effPlotterMediumMiniAOD*effPlotterTightMiniAOD)   

effPlotterLoose_Phase2=effPlotterLoose.clone()
effPlotterLoose_Phase2.vtxBin=20
effPlotterLoose_Phase2.vtxMin=149.5
effPlotterLoose_Phase2.vtxMax=249.5


effPlotterMedium_Phase2=effPlotterMedium.clone()                                                                                            
effPlotterMedium_Phase2.vtxBin=20                                                                                                           
effPlotterMedium_Phase2.vtxMin=149.5                                                                                                        
effPlotterMedium_Phase2.vtxMax=249.5 


effPlotterTight_Phase2=effPlotterTight.clone()                                                                                              
effPlotterTight_Phase2.vtxBin=20                                                                                                           
effPlotterTight_Phase2.vtxMin=149.5                                                                                                         
effPlotterTight_Phase2.vtxMax=249.5 

effPlotterLooseMiniAOD_Phase2=effPlotterLooseMiniAOD.clone()                                                                                
effPlotterLooseMiniAOD_Phase2.vtxBin=20                                                                                                     
effPlotterLooseMiniAOD_Phase2.vtxMin=149.5                                                                                                 
effPlotterLooseMiniAOD_Phase2.vtxMax=249.5                                                                                                

effPlotterMediumMiniAOD_Phase2=effPlotterMediumMiniAOD.clone()                                                                              
effPlotterMediumMiniAOD_Phase2.vtxBin=20                                                                                                   
effPlotterMediumMiniAOD_Phase2.vtxMin=149.5                                                                                                 
effPlotterMediumMiniAOD_Phase2.vtxMax=249.5                                                                                                

effPlotterTightMiniAOD_Phase2=effPlotterTightMiniAOD.clone()                                                                                
effPlotterTightMiniAOD_Phase2.vtxBin=20                                                                                                     
effPlotterTightMiniAOD_Phase2.vtxMin=149.5                                                                                                  
effPlotterTightMiniAOD_Phase2.vtxMax=249.5        

effPlotter_Phase2=cms.Sequence(effPlotterLoose_Phase2*effPlotterMedium_Phase2*effPlotterTight_Phase2)
effPlotter_miniAOD_Phase2=cms.Sequence(effPlotterLooseMiniAOD_Phase2*effPlotterMediumMiniAOD_Phase2*effPlotterTightMiniAOD_Phase2)                                    
                                                                                                             
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon                                                                         
phase2_muon.toReplaceWith(effPlotter,effPlotter_Phase2)     
phase2_muon.toReplaceWith(effPlotter_miniAOD,effPlotter_miniAOD_Phase2)  




