import FWCore.ParameterSet.Config as cms
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

MuonCosmicCompatibilityParameters = cms.PSet(
    CosmicCompFillerParameters = cms.PSet(
      MuonServiceProxy,
    
      InputTrackCollections = cms.VInputTag(cms.InputTag("generalTracks"), cms.InputTag("cosmicsVetoTracks")),
      InputMuonCollection = cms.InputTag("globalMuons"),
      InputCosmicMuonCollection = cms.InputTag("muonsFromCosmics1Leg"),
      InputVertexCollection = cms.InputTag("offlinePrimaryVertices"),
      
      # preselect |track.dxy+mu.dxy|<ipCut <== opp going tracks coming from
      # the same point have |track.dxy+mu.dxy|  d0Error. Thats as long as the charge is right
      
      ipCut = cms.double(0.1),
      # this is an inverted angle: back-to-back tracks have small value
      angleCut = cms.double(0.01),
      #currently not used
      deltaPhi = cms.double(0.1),
      # fractional difference in pt
      deltaPt = cms.double(0.05),
      
      # Time match
      offTimePos = cms.double(20.),
      offTimeNeg = cms.double(-20.),
      # min number of shared hits of mu with a cosmic mu
      sharedHits = cms.int32(5),
      # fraction of shared hits with a cosmic mu
      sharedFrac = cms.double(.75),
      
      # distance to PV: a match to PV makes it less cosm-like
      maxdzLoose = cms.double(5.),
      maxdxyLoose = cms.double(0.5),
      maxdzTight = cms.double(1.),
      maxdxyTight = cms.double(0.05),
      # vertex quality 
      minNDOF = cms.double(4),
      minvProb = cms.double(0.001)
      
      )
    )
