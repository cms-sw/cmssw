import FWCore.ParameterSet.Config as cms
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

MuonCosmicCompatibilityParameters = cms.PSet(
    CosmicCompFillerParameters = cms.PSet(
      MuonServiceProxy,
    
      InputTrackCollections = cms.VInputTag(cms.InputTag("generalTracks"), cms.InputTag("cosmicsVetoTracks")),
      InputMuonCollections = cms.VInputTag(cms.InputTag("globalMuons"), cms.InputTag("muons1stStep")),
      InputCosmicMuonCollection = cms.InputTag("muonsFromCosmics1Leg"),
      InputVertexCollection = cms.InputTag("offlinePrimaryVertices"),
      
      # preselect |track.dxy+mu.dxy|<ipCut <== opp going tracks coming from
      # the same point have |track.dxy+mu.dxy|  d0Error. Thats as long as the charge is right
    
    #used to find a b2b partner  
    angleCut = cms.double(0.1),
    deltaPt = cms.double(0.1),
    #timing
    offTimeNegLoose = cms.double(-15.),
    offTimePosLoose = cms.double(15.),
    offTimeNegTight = cms.double(-20.),
    offTimePosTight = cms.double(25.),
    offTimeNegLooseMult = cms.double(-15.),
    offTimePosLooseMult = cms.double(15.),
    offTimeNegTightMult = cms.double(-20.),
    offTimePosTightMult = cms.double(25.),
    corrTimeNeg = cms.double(-10),
    corrTimePos = cms.double(5),
    #shared hits
    sharedHits = cms.int32(5),
    sharedFrac = cms.double(.75),
    ipCut = cms.double(0.02),
    #ip, vertex
    maxdzLoose = cms.double(0.1),
    maxdxyLoose = cms.double(0.01),
    maxdzTight = cms.double(10.),
    maxdxyTight = cms.double(1.0),
    maxdzLooseMult = cms.double(0.1),
    maxdxyLooseMult = cms.double(0.01),
    maxdzTightMult = cms.double(10.),
    maxdxyTightMult = cms.double(1.0),
    hIpTrdxy = cms.double(0.02),
    hIpTrvProb = cms.double(0.5),
    largedxyMult = cms.double(3.0),
    largedxy = cms.double(2.0),
    minvProb = cms.double(0.001),
    maxvertZ = cms.double(20),
    maxvertRho = cms.double(5),
    nTrackThreshold = cms.int32(3),

    #muon id
    nChamberMatches = cms.int32(1),
    segmentComp = cms.double(0.4)

      )
    )
