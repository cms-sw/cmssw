import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

CosmicCompatibilityBlock = cms.PSet(
    CosmicCompFillerParameters = cms.PSet(
      MuonServiceProxy,
      
      InputTrackCollection = cms.InputTag("generalTracks"),
      InputMuonCollection = cms.InputTag("globalMuons"),
      InputCosmicCollection = cms.InputTag("muonsFromCosmics"),
      InputVertexCollection = cms.InputTag("offlinePrimaryVertices"),
      
      ipCut = cms.double(0.02),
      angleCut = cms.double(3.),
      deltaPhi = cms.double(0.1),
      deltaPt = cms.double(3.),
      offTimePos = cms.double(20.),
      offTimeNeg = cms.double(-20.),
      sharedHits = cms.int32(5),
      sharedFrac = cms.double(.75),
      maxdzLoose = cms.double(1.),
      maxdxyLoose = cms.double(0.05),
      maxdzTight = cms.double(5.),
      maxdxyTight = cms.double(0.5),
      minNDOF = cms.double(4),
      minvProb = cms.double(0.001),
      
      )
    )
