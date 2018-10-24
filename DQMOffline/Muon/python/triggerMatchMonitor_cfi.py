import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
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
                                              triggerPathName1 = cms.string('HLT_IsoMu24_v*'),
                                              triggerHistName1 = cms.string('IsoMu24'),
                                              triggerPtThresholdPath1 = cms.double(20.0),
                                              triggerPathName2 = cms.string('HLT_Mu50_v*'),
                                              triggerHistName2 = cms.string('Mu50'),
                                              triggerPtThresholdPath2 = cms.double(40.0),
                                              folder = cms.string("Muons_miniAOD/TriggerMatchMonitor")
                                              )
