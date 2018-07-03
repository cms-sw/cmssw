import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
MuonMiniAOD = DQMEDAnalyzer('MuonMiniAOD',
                             MuonServiceProxy,
                             MuonCollection       = cms.InputTag("slimmedMuons"),
                             VertexLabel     = cms.InputTag("offlineSlimmedPrimaryVertices"),
                             BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),
                             )
                             
