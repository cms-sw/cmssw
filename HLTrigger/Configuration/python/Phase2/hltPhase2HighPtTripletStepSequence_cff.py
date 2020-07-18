import FWCore.ParameterSet.Config as cms

hltPhase2HighPtTripletStepClusters = cms.EDProducer(
    "TrackClusterRemoverPhase2",
    TrackQuality=cms.string(""),
    maxChi2=cms.double(9.0),
    mightGet=cms.optional.untracked.vstring,
    minNumberOfLayersWithMeasBeforeFiltering=cms.int32(0),
    oldClusterRemovalInfo=cms.InputTag("hltPhase2PixelTrackClusters"),
    overrideTrkQuals=cms.InputTag(""),
    phase2OTClusters=cms.InputTag("siPhase2Clusters"),
    phase2pixelClusters=cms.InputTag("siPixelClusters"),
    trackClassifier=cms.InputTag("", "QualityMasks"),
    trajectories=cms.InputTag("hltPhase2InitialStepTracks"),
)

hltPhase2HighPtTripletStepSeedLayers = cms.EDProducer(
    "SeedingLayersEDProducer",
    BPix=cms.PSet(
        HitProducer=cms.string("siPixelRecHits"),
        TTRHBuilder=cms.string("WithTrackAngle"),
        skipClusters=cms.InputTag("hltPhase2HighPtTripletStepClusters"),
    ),
    FPix=cms.PSet(
        HitProducer=cms.string("siPixelRecHits"),
        TTRHBuilder=cms.string("WithTrackAngle"),
        skipClusters=cms.InputTag("hltPhase2HighPtTripletStepClusters"),
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
        "BPix1+BPix2+BPix3",
        "BPix2+BPix3+BPix4",
        "BPix1+BPix3+BPix4",
        "BPix1+BPix2+BPix4",
        "BPix2+BPix3+FPix1_pos",
        "BPix2+BPix3+FPix1_neg",
        "BPix1+BPix2+FPix1_pos",
        "BPix1+BPix2+FPix1_neg",
        "BPix2+FPix1_pos+FPix2_pos",
        "BPix2+FPix1_neg+FPix2_neg",
        "BPix1+FPix1_pos+FPix2_pos",
        "BPix1+FPix1_neg+FPix2_neg",
        "FPix1_pos+FPix2_pos+FPix3_pos",
        "FPix1_neg+FPix2_neg+FPix3_neg",
        "BPix1+FPix2_pos+FPix3_pos",
        "BPix1+FPix2_neg+FPix3_neg",
        "FPix2_pos+FPix3_pos+FPix4_pos",
        "FPix2_neg+FPix3_neg+FPix4_neg",
        "FPix3_pos+FPix4_pos+FPix5_pos",
        "FPix3_neg+FPix4_neg+FPix5_neg",
        "FPix4_pos+FPix5_pos+FPix6_pos",
        "FPix4_neg+FPix5_neg+FPix6_neg",
        "FPix5_pos+FPix6_pos+FPix7_pos",
        "FPix5_neg+FPix6_neg+FPix7_neg",
        "FPix6_pos+FPix7_pos+FPix8_pos",
        "FPix6_neg+FPix7_neg+FPix8_neg",
    ),
    mightGet=cms.optional.untracked.vstring,
)

hltPhase2HighPtTripletStepTrackingRegions = cms.EDProducer(
    "GlobalTrackingRegionFromBeamSpotEDProducer",
    RegionPSet=cms.PSet(
        beamSpot=cms.InputTag("offlineBeamSpot"),
        nSigmaZ=cms.double(4),
        originHalfLength=cms.double(0),
        originRadius=cms.double(0.02),
        precise=cms.bool(True),
        ptMin=cms.double(0.9),
        useMultipleScattering=cms.bool(False),
    ),
    mightGet=cms.optional.untracked.vstring,
)

hltPhase2HighPtTripletStepHitDoublets = cms.EDProducer(
    "HitPairEDProducer",
    clusterCheck=cms.InputTag("trackerClusterCheck"),
    layerPairs=cms.vuint32(0, 1),
    maxElement=cms.uint32(50000000),
    maxElementTotal=cms.uint32(50000000),
    mightGet=cms.optional.untracked.vstring,
    produceIntermediateHitDoublets=cms.bool(True),
    produceSeedingHitSets=cms.bool(False),
    seedingLayers=cms.InputTag("hltPhase2HighPtTripletStepSeedLayers"),
    trackingRegions=cms.InputTag("hltPhase2HighPtTripletStepTrackingRegions"),
    trackingRegionsSeedingLayers=cms.InputTag(""),
)

hltPhase2HighPtTripletStepHitTriplets = cms.EDProducer(
    "CAHitTripletEDProducer",
    CAHardPtCut=cms.double(0.5),
    CAPhiCut=cms.double(0.06),
    CAThetaCut=cms.double(0.003),
    SeedComparitorPSet=cms.PSet(
        ComponentName=cms.string("LowPtClusterShapeSeedComparitor"),
        clusterShapeCacheSrc=cms.InputTag("siPixelClusterShapeCache"),
        clusterShapeHitFilter=cms.string("ClusterShapeHitFilter"),
    ),
    doublets=cms.InputTag("hltPhase2HighPtTripletStepHitDoublets"),
    extraHitRPhitolerance=cms.double(0.032),
    maxChi2=cms.PSet(
        enabled=cms.bool(True),
        pt1=cms.double(0.8),
        pt2=cms.double(8),
        value1=cms.double(100),
        value2=cms.double(6),
    ),
    mightGet=cms.untracked.vstring(
        "IntermediateHitDoublets_highPtTripletStepHitDoublets__RECO",
        "IntermediateHitDoublets_hltPhase2HighPtTripletStepHitDoublets__RECO2",
    ),
    useBendingCorrection=cms.bool(True),
)

hltPhase2HighPtTripletStepSeeds = cms.EDProducer(
    "SeedCreatorFromRegionConsecutiveHitsEDProducer",
    MinOneOverPtError=cms.double(1),
    OriginTransverseErrorMultiplier=cms.double(1),
    SeedComparitorPSet=cms.PSet(ComponentName=cms.string("none")),
    SeedMomentumForBOFF=cms.double(5),
    TTRHBuilder=cms.string("WithTrackAngle"),
    forceKinematicWithRegionDirection=cms.bool(False),
    magneticField=cms.string(""),
    mightGet=cms.untracked.vstring(
        "RegionsSeedingHitSets_highPtTripletStepHitTriplets__RECO",
        "RegionsSeedingHitSets_hltPhase2HighPtTripletStepHitTriplets__RECO2",
    ),
    propagator=cms.string("PropagatorWithMaterial"),
    seedingHitSets=cms.InputTag("hltPhase2HighPtTripletStepHitTriplets"),
)

hltPhase2HighPtTripletStepTrackCandidates = cms.EDProducer(
    "CkfTrackCandidateMaker",
    MeasurementTrackerEvent=cms.InputTag("MeasurementTrackerEvent"),
    NavigationSchool=cms.string("SimpleNavigationSchool"),
    RedundantSeedCleaner=cms.string("CachingSeedCleanerBySharedInput"),
    SimpleMagneticField=cms.string("ParabolicMf"),
    TrajectoryBuilder=cms.string("GroupedCkfTrajectoryBuilder"),
    TrajectoryBuilderPSet=cms.PSet(
        refToPSet_=cms.string("hltPhase2HighPtTripletStepTrajectoryBuilder")
    ),
    TrajectoryCleaner=cms.string(
        "hltPhase2HighPtTripletStepTrajectoryCleanerBySharedHits"
    ),
    TransientInitialStateEstimatorParameters=cms.PSet(
        numberMeasurementsForFit=cms.int32(4),
        propagatorAlongTISE=cms.string("PropagatorWithMaterialParabolicMf"),
        propagatorOppositeTISE=cms.string("PropagatorWithMaterialParabolicMfOpposite"),
    ),
    cleanTrajectoryAfterInOut=cms.bool(True),
    doSeedingRegionRebuilding=cms.bool(True),
    maxNSeeds=cms.uint32(100000),
    maxSeedsBeforeCleaning=cms.uint32(1000),
    numHitsForSeedCleaner=cms.int32(50),
    onlyPixelHitsForSeedCleaner=cms.bool(True),
    phase2clustersToSkip=cms.InputTag("hltPhase2HighPtTripletStepClusters"),
    reverseTrajectories=cms.bool(False),
    src=cms.InputTag("hltPhase2HighPtTripletStepSeeds"),
    useHitsSplitting=cms.bool(False),
)

hltPhase2HighPtTripletStepTracks = cms.EDProducer(
    "TrackProducer",
    AlgorithmName=cms.string("highPtTripletStep"),
    Fitter=cms.string("FlexibleKFFittingSmoother"),
    GeometricInnerState=cms.bool(False),
    MeasurementTracker=cms.string(""),
    MeasurementTrackerEvent=cms.InputTag("MeasurementTrackerEvent"),
    NavigationSchool=cms.string("SimpleNavigationSchool"),
    Propagator=cms.string("RungeKuttaTrackerPropagator"),
    SimpleMagneticField=cms.string(""),
    TTRHBuilder=cms.string("WithTrackAngle"),
    TrajectoryInEvent=cms.bool(False),
    alias=cms.untracked.string("ctfWithMaterialTracks"),
    beamSpot=cms.InputTag("offlineBeamSpot"),
    clusterRemovalInfo=cms.InputTag(""),
    src=cms.InputTag("hltPhase2HighPtTripletStepTrackCandidates"),
    useHitsSplitting=cms.bool(False),
    useSimpleMF=cms.bool(False),
)

hltPhase2HighPtTripletStepTrackCutClassifier = cms.EDProducer(
    "TrackCutClassifier",
    beamspot=cms.InputTag("offlineBeamSpot"),
    ignoreVertices=cms.bool(False),
    mva=cms.PSet(
        dr_par=cms.PSet(
            d0err=cms.vdouble(0.003, 0.003, 0.003),
            d0err_par=cms.vdouble(0.002, 0.002, 0.001),
            dr_exp=cms.vint32(4, 4, 4),
            dr_par1=cms.vdouble(0.7, 0.6, 0.6),
            dr_par2=cms.vdouble(0.6, 0.5, 0.45),
        ),
        dz_par=cms.PSet(
            dz_exp=cms.vint32(4, 4, 4),
            dz_par1=cms.vdouble(0.8, 0.7, 0.7),
            dz_par2=cms.vdouble(0.6, 0.6, 0.55),
        ),
        maxChi2=cms.vdouble(9999.0, 9999.0, 9999.0),
        maxChi2n=cms.vdouble(2.0, 1.0, 0.8),
        maxDr=cms.vdouble(0.5, 0.03, 3.40282346639e38),
        maxDz=cms.vdouble(0.5, 0.2, 3.40282346639e38),
        maxDzWrtBS=cms.vdouble(3.40282346639e38, 24.0, 15.0),
        maxLostLayers=cms.vint32(3, 3, 2),
        min3DLayers=cms.vint32(3, 3, 4),
        minLayers=cms.vint32(3, 3, 4),
        minNVtxTrk=cms.int32(3),
        minNdof=cms.vdouble(1e-05, 1e-05, 1e-05),
        minPixelHits=cms.vint32(0, 0, 3),
    ),
    qualityCuts=cms.vdouble(-0.7, 0.1, 0.7),
    src=cms.InputTag("hltPhase2HighPtTripletStepTracks"),
    vertices=cms.InputTag("hltPhase2PixelVertices"),
)

hltPhase2HighPtTripletStepTrackSelectionHighPurity = cms.EDProducer(
    "TrackCollectionFilterCloner",
    copyExtras=cms.untracked.bool(True),
    copyTrajectories=cms.untracked.bool(False),
    minQuality=cms.string("highPurity"),
    originalMVAVals=cms.InputTag(
        "hltPhase2HighPtTripletStepTrackCutClassifier", "MVAValues"
    ),
    originalQualVals=cms.InputTag(
        "hltPhase2HighPtTripletStepTrackCutClassifier", "QualityMasks"
    ),
    originalSource=cms.InputTag("hltPhase2HighPtTripletStepTracks"),
)

hltPhase2HighPtTripletStepSequence = cms.Sequence(
    hltPhase2HighPtTripletStepClusters
    + hltPhase2HighPtTripletStepSeedLayers
    + hltPhase2HighPtTripletStepTrackingRegions
    + hltPhase2HighPtTripletStepHitDoublets
    + hltPhase2HighPtTripletStepHitTriplets
    + hltPhase2HighPtTripletStepSeeds
    + hltPhase2HighPtTripletStepTrackCandidates
    + hltPhase2HighPtTripletStepTracks
    + hltPhase2HighPtTripletStepTrackCutClassifier
    + hltPhase2HighPtTripletStepTrackSelectionHighPurity
)
