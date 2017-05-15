import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

effPlotter_Loose = DQMEDHarvester("EfficiencyPlotter",
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
                                  vtxBin = cms.int32(10),
                                  vtxMin = cms.double(0.5),
                                  vtxMax = cms.double(40.5),
                                  MuonID = cms.string("Loose")
                                  )


effPlotter_Medium = DQMEDHarvester("EfficiencyPlotter",
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
                                   vtxBin = cms.int32(10),
                                   vtxMin = cms.double(0.5),
                                   vtxMax = cms.double(40.5),
                                   MuonID = cms.string("Medium")
                                   )


effPlotter_Tight = DQMEDHarvester("EfficiencyPlotter",
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
                                  vtxBin = cms.int32(10),
                                  vtxMin = cms.double(0.5),
                                  vtxMax = cms.double(40.5),
                                  MuonID = cms.string("Tight")
                                  )
effPlotter_Loose_miniAOD = DQMEDHarvester("EfficiencyPlotter",
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
                                          vtxBin = cms.int32(10),
                                          vtxMin = cms.double(0.5),
                                          vtxMax = cms.double(40.5),
                                          MuonID = cms.string("Loose")
                                          )


effPlotter_Medium_miniAOD = DQMEDHarvester("EfficiencyPlotter",
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
                                           vtxBin = cms.int32(10),
                                           vtxMin = cms.double(0.5),
                                           vtxMax = cms.double(40.5),
                                           MuonID = cms.string("Medium")
                                           )


effPlotter_Tight_miniAOD = DQMEDHarvester("EfficiencyPlotter",
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
                                          vtxBin = cms.int32(10),
                                          vtxMin = cms.double(0.5),
                                          vtxMax = cms.double(40.5),
                                          MuonID = cms.string("Tight")
                                          )


