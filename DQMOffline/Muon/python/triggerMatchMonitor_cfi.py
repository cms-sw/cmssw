import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
triggerMatchMonitor = DQMEDAnalyzer('TriggerMatchMonitor',
                                      MuonServiceProxy,
                                      
                                      MuonCollection  = cms.InputTag("muons"),
                                      VertexLabel     = cms.InputTag("offlinePrimaryVertices"),
                                      BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),
                                   
                                      etaBin = cms.int32(100),
                                      etaMin = cms.double(-3.0),
                                      etaMax = cms.double(3.0),
                                      
                                      folder = cms.string("Muons/TriggerMatchMonitor")
                                      )
triggerMatchMonitor_miniAOD = DQMEDAnalyzer('TriggerMatchMonitor',
                                              MuonServiceProxy,
                                              offlineBeamSpot = cms.untracked.InputTag("offlineBeamSpot"),
                                              offlinePrimaryVertices = cms.untracked.InputTag("offlinePrimaryVertices"),
                                              MuonCollection  = cms.InputTag("slimmedMuons"),
                                              patMuonCollection  = cms.InputTag("slimmedMuons"),
                                              VertexLabel     = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                              BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),
                                              triggerResults = cms.untracked.InputTag("TriggerResults","","HLT"),
                                              triggerObjects = cms.InputTag("slimmedPatTrigger"),
                                              hltCollectionFilters = cms.vstring('*'),
                                              triggerNameList = cms.vstring('HLT_Mu27_v*'),
                                              folder = cms.string("Muons_miniAOD/TriggerMatchMonitor")
                                              )
