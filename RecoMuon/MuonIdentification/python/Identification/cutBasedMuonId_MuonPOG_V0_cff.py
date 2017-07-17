import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

cutBasedMuonId_MuonPOG_V0_loose = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-loose"),
    isPOGApproved = cms.untracked.bool(False),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("MuonTypeByOrCut"),
                  types = cms.vstring("PFMuon"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonTypeByOrCut"),
                  types = cms.vstring("GlobalMuon", "TrackerMuon"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
    )
)

cutBasedMuonId_MuonPOG_V0_medium = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-medium"),
    isPOGApproved = cms.untracked.bool(False),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("MuonTypeByOrCut"),
                  types = cms.vstring("PFMuon"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonTypeByOrCut"),
                  types = cms.vstring("GlobalMuon", "TrackerMuon"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonTrackCut"),
                  innerTrack = cms.PSet(
                      minValidFraction = cms.double(0.8), ),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonSegmentCompatibilityCut"),
                  goodGLB = cms.PSet(
                      maxGlbNormChi2 = cms.double(3.0),
                      maxChi2LocalPos = cms.double(12.0),
                      maxTrkKink = cms.double(20.0),
                  ),
                  minCompatGlb = cms.double(0.303),
                  minCompatNonGlb = cms.double(0.451),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
    )
)

cutBasedMuonId_MuonPOG_V0_tight = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-tight"),
    isPOGApproved = cms.untracked.bool(False),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("MuonTypeByOrCut"),
                  types = cms.vstring("PFMuon"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonTypeByOrCut"),
                  types = cms.vstring("GlobalMuon"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("GlobalMuonPromptTightCut"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonTrackCut"),
                  innerTrack = cms.PSet(
                      minTrackerLayersWithMeasurement = cms.int32(6),
                      minNumberOfValidPixelHits = cms.int32(1) ),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonMatchCut"),
                  minNumberOfMatchedStations = cms.int32(2),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonDxyCut"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  vertexSrcMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
                  trackType = cms.string("muonBestTrack"),
                  maxDxy = cms.double(0.2),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonDzCut"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  vertexSrcMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
                  trackType = cms.string("muonBestTrack"),
                  maxDz = cms.double(0.5),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
    )
)

cutBasedMuonId_MuonPOG_V0_soft = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-soft"),
    isPOGApproved = cms.untracked.bool(False),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("TMOneStationTightCut"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonTrackCut"),
                  innerTrack = cms.PSet(
                      minTrackerLayersWithMeasurement = cms.int32(6),
                      minPixelLayersWithMeasurement = cms.int32(1),
                      trackQuality = cms.string("highPurity") ),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonDxyCut"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  vertexSrcMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
                  trackType = cms.string("innerTrack"),
                  maxDxy = cms.double(0.3),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonDzCut"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  vertexSrcMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
                  trackType = cms.string("innerTrack"),
                  maxDz = cms.double(20.0),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
    )
)

cutBasedMuonId_MuonPOG_V0_highpt = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-highpt"),
    isPOGApproved = cms.untracked.bool(True),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("MuonTypeByOrCut"),
                  types = cms.vstring("GlobalMuon"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonTrackCut"),
                  innerTrack = cms.PSet(
                      minTrackerLayersWithMeasurement = cms.int32(6),
                      minNumberOfValidPixelHits = cms.int32(1) ),
                  globalTrack = cms.PSet(minNumberOfValidMuonHits = cms.int32(1) ),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonMatchCut"),
                  minNumberOfMatchedStations = cms.int32(2),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonMomQualityCut"),
                  maxRelPtErr = cms.double(0.3),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonDxyCut"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  vertexSrcMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
                  trackType = cms.string("muonBestTrack"),
                  maxDxy = cms.double(0.2),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("MuonDzCut"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  vertexSrcMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
                  trackType = cms.string("muonBestTrack"),
                  maxDz = cms.double(0.5),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
    )
)

central_id_registry.register(cutBasedMuonId_MuonPOG_V0_loose.idName , 'd19c494fb8227d7af3c8e29053b1934a')
central_id_registry.register(cutBasedMuonId_MuonPOG_V0_medium.idName, '1f4bb781e8d98b4cb281de5c9b3fd193')
central_id_registry.register(cutBasedMuonId_MuonPOG_V0_tight.idName , '4c815640b5477c514210d7cdbde98fe0')
central_id_registry.register(cutBasedMuonId_MuonPOG_V0_soft.idName  , '12da9c6aae4d389980da3f463aac2afb')
central_id_registry.register(cutBasedMuonId_MuonPOG_V0_highpt.idName, '3f6a009a63dc9eb6af3f2de17d53c9fd')

