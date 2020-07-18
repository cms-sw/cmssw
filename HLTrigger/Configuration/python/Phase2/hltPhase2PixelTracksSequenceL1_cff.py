import FWCore.ParameterSet.Config as cms

hltPhase2PixelTrackClusters = cms.EDProducer(
    "TrackClusterRemoverPhase2",
    TrackQuality=cms.string(" "),
    maxChi2=cms.double(9.0),
    mightGet=cms.optional.untracked.vstring,
    minNumberOfLayersWithMeasBeforeFiltering=cms.int32(0),
    oldClusterRemovalInfo=cms.InputTag(""),
    overrideTrkQuals=cms.InputTag(""),
    phase2OTClusters=cms.InputTag("siPhase2Clusters"),
    phase2pixelClusters=cms.InputTag("siPixelClusters"),
    trackClassifier=cms.InputTag("", "QualityMasks"),
    trajectories=cms.InputTag("hltPhase2L1CtfTracks"),
)

hltPhase2PixelTrackFilterByKinematics = cms.EDProducer(
    "PixelTrackFilterByKinematicsProducer",
    chi2=cms.double(1000.0),
    nSigmaInvPtTolerance=cms.double(0.0),
    nSigmaTipMaxTolerance=cms.double(0.0),
    ptMin=cms.double(0.9),
    tipMax=cms.double(1.0),
)

hltPhase2PixelFitterByHelixProjections = cms.EDProducer(
    "PixelFitterByHelixProjectionsProducer",
    scaleErrorsForBPix1=cms.bool(False),
    scaleFactor=cms.double(0.65),
)

hltPhase2PixelTracksTrackingRegions = cms.EDProducer(
    "GlobalTrackingRegionFromBeamSpotEDProducer",
    RegionPSet=cms.PSet(
        beamSpot=cms.InputTag("offlineBeamSpot"),
        nSigmaZ=cms.double(4.0),
        originRadius=cms.double(0.02),
        precise=cms.bool(True),
        ptMin=cms.double(0.9),
    ),
)

hltPhase2PixelTracksSeedLayers = cms.EDProducer(
    "SeedingLayersEDProducer",
    BPix=cms.PSet(
        HitProducer=cms.string("siPixelRecHits"),
        TTRHBuilder=cms.string("WithTrackAngle"),
        skipClusters=cms.InputTag("hltPhase2PixelTrackClusters"),
    ),
    FPix=cms.PSet(
        HitProducer=cms.string("siPixelRecHits"),
        TTRHBuilder=cms.string("WithTrackAngle"),
        skipClusters=cms.InputTag("hltPhase2PixelTrackClusters"),
    ),
    MTEC=cms.PSet(),
    MTIB=cms.PSet(),
    MTID=cms.PSet(),
    MTOB=cms.PSet(),
    TEC=cms.PSet(),
    TIB=cms.PSet(),
    TID=cms.PSet(),
    TOB=cms.PSet(),
    layerList=cms.vstring(
        "BPix1+BPix2+BPix3+BPix4",
        "BPix1+BPix2+BPix3+FPix1_pos",
        "BPix1+BPix2+BPix3+FPix1_neg",
        "BPix1+BPix2+FPix1_pos+FPix2_pos",
        "BPix1+BPix2+FPix1_neg+FPix2_neg",
        "BPix1+FPix1_pos+FPix2_pos+FPix3_pos",
        "BPix1+FPix1_neg+FPix2_neg+FPix3_neg",
        "FPix1_pos+FPix2_pos+FPix3_pos+FPix4_pos",
        "FPix1_neg+FPix2_neg+FPix3_neg+FPix4_neg",
        "FPix2_pos+FPix3_pos+FPix4_pos+FPix5_pos",
        "FPix2_neg+FPix3_neg+FPix4_neg+FPix5_neg",
        "FPix3_pos+FPix4_pos+FPix5_pos+FPix6_pos",
        "FPix3_neg+FPix4_neg+FPix5_neg+FPix6_neg",
        "FPix4_pos+FPix5_pos+FPix6_pos+FPix7_pos",
        "FPix4_neg+FPix5_neg+FPix6_neg+FPix7_neg",
        "FPix5_pos+FPix6_pos+FPix7_pos+FPix8_pos",
        "FPix5_neg+FPix6_neg+FPix7_neg+FPix8_neg",
    ),
)

hltPhase2PixelTracksHitDoublets = cms.EDProducer(
    "HitPairEDProducer",
    clusterCheck=cms.InputTag(""),
    layerPairs=cms.vuint32(0, 1, 2),
    maxElement=cms.uint32(50000000),
    maxElementTotal=cms.uint32(50000000),
    produceIntermediateHitDoublets=cms.bool(True),
    produceSeedingHitSets=cms.bool(False),
    seedingLayers=cms.InputTag("hltPhase2PixelTracksSeedLayers"),
    trackingRegions=cms.InputTag("hltPhase2PixelTracksTrackingRegions"),
    trackingRegionsSeedingLayers=cms.InputTag(""),
)

hltPhase2PixelTracksHitSeeds = cms.EDProducer(
    "CAHitQuadrupletEDProducer",
    CAHardPtCut=cms.double(0.0),
    CAPhiCut=cms.double(0.2),
    CAThetaCut=cms.double(0.0012),
    SeedComparitorPSet=cms.PSet(
        ComponentName=cms.string("LowPtClusterShapeSeedComparitor"),
        clusterShapeCacheSrc=cms.InputTag("siPixelClusterShapeCache"),
        clusterShapeHitFilter=cms.string("ClusterShapeHitFilter"),
    ),
    doublets=cms.InputTag("hltPhase2PixelTracksHitDoublets"),
    extraHitRPhitolerance=cms.double(0.032),
    fitFastCircle=cms.bool(True),
    fitFastCircleChi2Cut=cms.bool(True),
    maxChi2=cms.PSet(
        enabled=cms.bool(True),
        pt1=cms.double(0.7),
        pt2=cms.double(2.0),
        value1=cms.double(200.0),
        value2=cms.double(50.0),
    ),
    mightGet=cms.untracked.vstring(
        "IntermediateHitDoublets_hltPhase2PixelTracksHitDoublets__RECO2"
    ),
    useBendingCorrection=cms.bool(True),
)

hltPhase2PixelTracks = cms.EDProducer(
    "PixelTrackProducer",
    Cleaner=cms.string("hltPhase2PixelTrackCleanerBySharedHits"),
    Filter=cms.InputTag("hltPhase2PixelTrackFilterByKinematics"),
    Fitter=cms.InputTag("hltPhase2PixelFitterByHelixProjections"),
    SeedingHitSets=cms.InputTag("hltPhase2PixelTracksHitSeeds"),
    mightGet=cms.untracked.vstring(
        "", "RegionsSeedingHitSets_hltPhase2PixelTracksHitSeeds__RECO2"
    ),
    passLabel=cms.string("hltPhase2PixelTracks"),
)

hltPhase2PixelTracksSequenceL1 = cms.Sequence(
    hltPhase2PixelTrackClusters
    + hltPhase2PixelTrackFilterByKinematics
    + hltPhase2PixelFitterByHelixProjections
    + hltPhase2PixelTracksTrackingRegions
    + hltPhase2PixelTracksSeedLayers
    + hltPhase2PixelTracksHitDoublets
    + hltPhase2PixelTracksHitSeeds
    + hltPhase2PixelTracks
)
