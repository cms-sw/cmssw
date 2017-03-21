import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *


MuonMiniAOD = cms.EDAnalyzer("MuonMiniAOD",
                             MuonServiceProxy,
                             MuonCollection       = cms.InputTag("slimmedMuons"),
                             VertexLabel     = cms.InputTag("offlineSlimmedPrimaryVertices"),
                             BeamSpotLabel   = cms.InputTag("offlineBeamSpot"),
                             )
                             
