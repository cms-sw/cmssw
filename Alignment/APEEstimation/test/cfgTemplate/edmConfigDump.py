Input sample:  data1
Input file number 1
Iteration number:  0
Last iteration:  False
AlignmentRcd:  design
import FWCore.ParameterSet.Config as cms

process = cms.Process("ApeEstimator")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/a/ajkumar/APE_newCPE/CMSSW_7_2_0_pre6/src/apeSkim_1_1_0X0.root'),
    duplicateCheckMode = cms.untracked.string('checkEachRealDataFile')
)
process.MeasurementTrackerEvent = cms.EDProducer("MeasurementTrackerEventProducer",
    inactivePixelDetectorLabels = cms.VInputTag(),
    stripClusterProducer = cms.string('MuSkim'),
    pixelClusterProducer = cms.string('MuSkim'),
    switchOffPixelsIfEmpty = cms.bool(True),
    inactiveStripDetectorLabels = cms.VInputTag(),
    skipClusters = cms.InputTag(""),
    measurementTracker = cms.string('')
)


process.MixedLayerPairs = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2', 
        'BPix1+BPix3', 
        'BPix2+BPix3', 
        'BPix1+FPix1_pos', 
        'BPix1+FPix1_neg', 
        'BPix1+FPix2_pos', 
        'BPix1+FPix2_neg', 
        'BPix2+FPix1_pos', 
        'BPix2+FPix1_neg', 
        'BPix2+FPix2_pos', 
        'BPix2+FPix2_neg', 
        'FPix1_pos+FPix2_pos', 
        'FPix1_neg+FPix2_neg', 
        'FPix2_pos+TEC1_pos', 
        'FPix2_pos+TEC2_pos', 
        'TEC1_pos+TEC2_pos', 
        'TEC2_pos+TEC3_pos', 
        'FPix2_neg+TEC1_neg', 
        'FPix2_neg+TEC2_neg', 
        'TEC1_neg+TEC2_neg', 
        'TEC2_neg+TEC3_neg'),
    MTOB = cms.PSet(

    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(1)
    ),
    MTID = cms.PSet(

    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    MTEC = cms.PSet(

    ),
    MTIB = cms.PSet(

    ),
    TID = cms.PSet(

    ),
    TOB = cms.PSet(

    ),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    TIB = cms.PSet(

    )
)


process.MixedLayerTriplets = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg', 
        'BPix1+BPix2+TIB1', 
        'BPix1+BPix3+TIB1', 
        'BPix2+BPix3+TIB1', 
        'BPix1+FPix1_pos+TID1_pos', 
        'BPix1+FPix1_neg+TID1_neg', 
        'BPix1+FPix1_pos+TID2_pos', 
        'BPix1+FPix1_neg+TID2_neg', 
        'FPix1_pos+FPix2_pos+TEC1_pos', 
        'FPix1_neg+FPix2_neg+TEC1_neg', 
        'FPix1_pos+FPix2_pos+TEC2_pos', 
        'FPix1_neg+FPix2_neg+TEC2_neg'),
    MTOB = cms.PSet(

    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    MTID = cms.PSet(

    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    MTEC = cms.PSet(

    ),
    MTIB = cms.PSet(

    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB = cms.PSet(

    ),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)


process.PixelLayerPairs = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2', 
        'BPix1+BPix3', 
        'BPix2+BPix3', 
        'BPix1+FPix1_pos', 
        'BPix1+FPix1_neg', 
        'BPix1+FPix2_pos', 
        'BPix1+FPix2_neg', 
        'BPix2+FPix1_pos', 
        'BPix2+FPix1_neg', 
        'BPix2+FPix2_pos', 
        'BPix2+FPix2_neg', 
        'FPix1_pos+FPix2_pos', 
        'FPix1_neg+FPix2_neg'),
    MTOB = cms.PSet(

    ),
    TEC = cms.PSet(

    ),
    MTID = cms.PSet(

    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    MTEC = cms.PSet(

    ),
    MTIB = cms.PSet(

    ),
    TID = cms.PSet(

    ),
    TOB = cms.PSet(

    ),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    TIB = cms.PSet(

    )
)


process.PixelLayerTriplets = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg'),
    MTOB = cms.PSet(

    ),
    TEC = cms.PSet(

    ),
    MTID = cms.PSet(

    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    MTEC = cms.PSet(

    ),
    MTIB = cms.PSet(

    ),
    TID = cms.PSet(

    ),
    TOB = cms.PSet(

    ),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    TIB = cms.PSet(

    )
)


process.TrackProducer = cms.EDProducer("TrackProducer",
    src = cms.InputTag("ckfTrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    GeometricInnerState = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    AlgorithmName = cms.string('undefAlgorithm'),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.TrackRefitter = cms.EDProducer("TrackRefitter",
    src = cms.InputTag("generalTracks"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    srcConstr = cms.InputTag(""),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    constraint = cms.string(''),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.TrackRefitterBHM = cms.EDProducer("TrackRefitter",
    src = cms.InputTag("ctfWithMaterialTracksBeamHaloMuon"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    srcConstr = cms.InputTag(""),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('KFFittingSmootherBH'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    constraint = cms.string(''),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(True),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('BeamHaloPropagatorAlong')
)


process.TrackRefitterForApeEstimator = cms.EDProducer("TrackRefitter",
    src = cms.InputTag("MuSkim"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    srcConstr = cms.InputTag(""),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    constraint = cms.string(''),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.TrackRefitterHighPurityForApeEstimator = cms.EDProducer("TrackRefitter",
    src = cms.InputTag("HighPuritySelector"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    srcConstr = cms.InputTag(""),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    constraint = cms.string(''),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.TrackRefitterP5 = cms.EDProducer("TrackRefitter",
    src = cms.InputTag("ctfWithMaterialTracksP5"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    srcConstr = cms.InputTag(""),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('FittingSmootherRKP5'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    constraint = cms.string(''),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(True),
    AlgorithmName = cms.string('ctf'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.ak4CaloJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    doAreaFastjet = cms.bool(False),
    Ghost_EtaMax = cms.double(5.0),
    doAreaDiskApprox = cms.bool(False),
    jetType = cms.string('CaloJet'),
    minSeed = cms.uint32(14327),
    voronoiRfact = cms.double(-0.9),
    doRhoFastjet = cms.bool(False),
    nSigmaPU = cms.double(1.0),
    GhostArea = cms.double(0.01),
    Rho_EtaMax = cms.double(4.4),
    useDeterministicSeed = cms.bool(True),
    doPVCorrection = cms.bool(True),
    doOutputJets = cms.bool(True),
    src = cms.InputTag("towerMaker"),
    inputEtMin = cms.double(0.3),
    puPtMin = cms.double(10),
    srcPVs = cms.InputTag("offlinePrimaryVertices"),
    jetPtMin = cms.double(3.0),
    radiusPU = cms.double(0.5),
    doPUOffsetCorr = cms.bool(False),
    inputEMin = cms.double(0.0),
    maxRecoveredHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxBadEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    jetAlgorithm = cms.string('AntiKt'),
    rParam = cms.double(0.4)
)


process.ak4CaloJetsForTrk = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    doAreaFastjet = cms.bool(False),
    voronoiRfact = cms.double(-0.9),
    maxBadHcalCells = cms.uint32(9999999),
    doAreaDiskApprox = cms.bool(False),
    maxRecoveredEcalCells = cms.uint32(9999999),
    jetType = cms.string('CaloJet'),
    minSeed = cms.uint32(14327),
    Ghost_EtaMax = cms.double(5.0),
    doRhoFastjet = cms.bool(False),
    jetAlgorithm = cms.string('AntiKt'),
    nSigmaPU = cms.double(1.0),
    GhostArea = cms.double(0.01),
    Rho_EtaMax = cms.double(4.4),
    maxBadEcalCells = cms.uint32(9999999),
    useDeterministicSeed = cms.bool(True),
    doPVCorrection = cms.bool(True),
    maxRecoveredHcalCells = cms.uint32(9999999),
    rParam = cms.double(0.4),
    maxProblematicHcalCells = cms.uint32(9999999),
    doOutputJets = cms.bool(True),
    src = cms.InputTag("caloTowerForTrk"),
    inputEtMin = cms.double(0.3),
    puPtMin = cms.double(10),
    srcPVs = cms.InputTag("pixelVertices"),
    jetPtMin = cms.double(3.0),
    radiusPU = cms.double(0.5),
    maxProblematicEcalCells = cms.uint32(9999999),
    doPUOffsetCorr = cms.bool(False),
    inputEMin = cms.double(0.0)
)


process.beamhaloTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("beamhaloTrackerSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('BeamHaloPropagatorAlong'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('BeamHaloPropagatorOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('CkfTrajectoryBuilderBeamHalo')
    ),
    NavigationSchool = cms.string('BeamHaloNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder')
)


process.beamhaloTrackerSeedingLayers = cms.EDProducer("SeedingLayersEDProducer",
    TEC4 = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    TEC5 = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    TEC6 = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    TEC = cms.PSet(
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(7)
    ),
    TEC1 = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    TEC2 = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    TEC3 = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    layerList = cms.vstring('FPix1_pos+FPix2_pos', 
        'FPix1_neg+FPix2_neg', 
        'TID2_pos+TID3_pos', 
        'TID2_neg+TID3_neg', 
        'TEC1_neg+TEC2_neg', 
        'TEC1_pos+TEC2_pos', 
        'TEC2_neg+TEC3_neg', 
        'TEC2_pos+TEC3_pos', 
        'TEC3_neg+TEC4_neg', 
        'TEC3_pos+TEC4_pos', 
        'TEC4_neg+TEC5_neg', 
        'TEC4_pos+TEC5_pos', 
        'TEC5_neg+TEC6_neg', 
        'TEC5_pos+TEC6_pos', 
        'TEC7_neg+TEC8_neg', 
        'TEC7_pos+TEC8_pos', 
        'TEC8_neg+TEC9_neg', 
        'TEC8_pos+TEC9_pos')
)


process.beamhaloTrackerSeeds = cms.EDProducer("CtfSpecialSeedGenerator",
    ErrorRescaling = cms.double(50.0),
    OrderedHitsFactoryPSets = cms.VPSet(cms.PSet(
        ComponentName = cms.string('BeamHaloPairGenerator'),
        maxTheta = cms.double(0.1),
        LayerSrc = cms.InputTag("beamhaloTrackerSeedingLayers"),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('outsideIn')
    ), 
        cms.PSet(
            ComponentName = cms.string('BeamHaloPairGenerator'),
            maxTheta = cms.double(0.1),
            LayerSrc = cms.InputTag("beamhaloTrackerSeedingLayers"),
            PropagationDirection = cms.string('oppositeToMomentum'),
            NavigationDirection = cms.string('outsideIn')
        )),
    Charges = cms.vint32(-1, 1),
    PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
    MaxNumberOfCosmicClusters = cms.uint32(10000),
    UseScintillatorsConstraint = cms.bool(False),
    SetMomentum = cms.bool(True),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(21.2),
            originZPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.9),
            originXPos = cms.double(0.0),
            originRadius = cms.double(0.2)
        ),
        ComponentName = cms.string('GlobalRegionProducer')
    ),
    SeedsFromNegativeY = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    doClusterCheck = cms.bool(True),
    SeedsFromPositiveY = cms.bool(False),
    MaxNumberOfPixelClusters = cms.uint32(10000),
    SeedMomentum = cms.double(15.0),
    maxSeeds = cms.int32(10000),
    CheckHitsAreOnDifferentLayers = cms.bool(False),
    ClusterCollectionLabel = cms.InputTag("siStripClusters"),
    requireBOFF = cms.bool(False)
)


process.beamhaloTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("beamhaloTrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('KFFittingSmootherBH'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('beamhalo'),
    alias = cms.untracked.string('beamhaloTracks'),
    NavigationSchool = cms.string('BeamHaloNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    GeometricInnerState = cms.bool(True),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('BeamHaloPropagatorAlong')
)


process.caloTowerForTrk = cms.EDProducer("CaloTowersCreator",
    EBSumThreshold = cms.double(0.2),
    MomHBDepth = cms.double(0.2),
    UseEtEBTreshold = cms.bool(False),
    hfInput = cms.InputTag("hfreco"),
    AllowMissingInputs = cms.bool(False),
    MomEEDepth = cms.double(0.0),
    EESumThreshold = cms.double(0.45),
    HBGrid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32(9999),
    HBThreshold = cms.double(0.7),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(),
    UseEcalRecoveredHits = cms.bool(False),
    MomConstrMethod = cms.int32(1),
    MomHEDepth = cms.double(0.4),
    HcalThreshold = cms.double(-1000.0),
    HF2Weights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HOWeights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    EEGrid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    UseSymEBTreshold = cms.bool(True),
    EEWeights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(True),
    HBWeights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HF1Weight = cms.double(1.0),
    HF2Grid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    HEDWeights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HEDGrid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    EBWeight = cms.double(1.0),
    HF1Grid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    EBWeights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    HESThreshold = cms.double(0.8),
    hbheInput = cms.InputTag("hbheprereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(0.85),
    HcalAcceptSeverityLevel = cms.uint32(9),
    EEThreshold = cms.double(0.3),
    HOThresholdPlus1 = cms.double(3.5),
    HOThresholdPlus2 = cms.double(3.5),
    HF1Weights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    hoInput = cms.InputTag("horeco"),
    HF1Threshold = cms.double(0.5),
    HOThresholdMinus1 = cms.double(3.5),
    HESGrid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    EcutTower = cms.double(-1000.0),
    UseRejectedRecoveredEcalHits = cms.bool(False),
    UseEtEETreshold = cms.bool(False),
    HESWeights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring('kTime', 
        'kWeird', 
        'kBad'),
    HEDWeight = cms.double(1.0),
    UseSymEETreshold = cms.bool(True),
    HEDThreshold = cms.double(0.8),
    EBThreshold = cms.double(0.07),
    UseRejectedHitsOnly = cms.bool(False),
    UseHcalRecoveredHits = cms.bool(True),
    HOThresholdMinus2 = cms.double(3.5),
    HOThreshold0 = cms.double(1.1),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    UseRejectedRecoveredHcalHits = cms.bool(True),
    MomEBDepth = cms.double(0.3),
    HBWeight = cms.double(1.0),
    HOGrid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    EBGrid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0)
)


process.calotowermaker = cms.EDProducer("CaloTowersCreator",
    EBSumThreshold = cms.double(0.2),
    MomHBDepth = cms.double(0.2),
    UseEtEBTreshold = cms.bool(False),
    hfInput = cms.InputTag("hfreco"),
    AllowMissingInputs = cms.bool(False),
    MomEEDepth = cms.double(0.0),
    EESumThreshold = cms.double(0.45),
    HBGrid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    HF1Weight = cms.double(1.0),
    HBThreshold = cms.double(0.7),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(),
    UseEcalRecoveredHits = cms.bool(False),
    MomConstrMethod = cms.int32(1),
    MomHEDepth = cms.double(0.4),
    HcalThreshold = cms.double(-1000.0),
    HF2Weights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HOWeights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    EEGrid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    UseSymEBTreshold = cms.bool(True),
    EEWeights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(True),
    HBWeights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HESWeight = cms.double(1.0),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32(9999),
    HF2Grid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    HEDWeights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HOThresholdPlus1 = cms.double(3.5),
    EBWeight = cms.double(1.0),
    HF1Grid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    EBWeights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HOWeight = cms.double(1.0),
    EBThreshold = cms.double(0.07),
    hbheInput = cms.InputTag("hbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(0.85),
    HcalAcceptSeverityLevel = cms.uint32(9),
    EEThreshold = cms.double(0.3),
    HESThreshold = cms.double(0.8),
    HOThresholdPlus2 = cms.double(3.5),
    HF1Weights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    hoInput = cms.InputTag("horeco"),
    HF1Threshold = cms.double(0.5),
    HOThresholdMinus1 = cms.double(3.5),
    HESGrid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    UseRejectedRecoveredEcalHits = cms.bool(False),
    UseEtEETreshold = cms.bool(False),
    HEDWeight = cms.double(1.0),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring('kTime', 
        'kWeird', 
        'kBad'),
    HESWeights = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    UseSymEETreshold = cms.bool(True),
    HEDThreshold = cms.double(0.8),
    UseRejectedHitsOnly = cms.bool(False),
    EcutTower = cms.double(-1000.0),
    HEDGrid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    UseHcalRecoveredHits = cms.bool(True),
    HOThresholdMinus2 = cms.double(3.5),
    HOThreshold0 = cms.double(1.1),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    UseRejectedRecoveredHcalHits = cms.bool(True),
    MomEBDepth = cms.double(0.3),
    HBWeight = cms.double(1.0),
    HOGrid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    EBGrid = cms.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0)
)


process.ckfTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("globalMixedSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('GroupedCkfTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder')
)


process.ckfTrackCandidatesCombinedSeeds = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("globalCombinedSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('GroupedCkfTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder')
)


process.ckfTrackCandidatesNoOverlaps = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("globalMixedSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('CkfTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder')
)


process.ckfTrackCandidatesP5 = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("combinedP5SeedsForCTF"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('GroupedCkfTrajectoryBuilderP5')
    ),
    NavigationSchool = cms.string('CosmicNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder')
)


process.ckfTrackCandidatesP5Bottom = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("combinedP5SeedsForCTFBottom"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('GroupedCkfTrajectoryBuilderP5Bottom')
    ),
    NavigationSchool = cms.string('CosmicNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder')
)


process.ckfTrackCandidatesP5LHCNavigation = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("combinedP5SeedsForCTF"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('GroupedCkfTrajectoryBuilderP5')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder')
)


process.ckfTrackCandidatesP5Top = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("combinedP5SeedsForCTFTop"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('GroupedCkfTrajectoryBuilderP5Top')
    ),
    NavigationSchool = cms.string('CosmicNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder')
)


process.ckfTrackCandidatesPixelLess = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("globalPixelLessSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('GroupedCkfTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder')
)


process.combinatorialcosmicseedfinder = cms.EDProducer("CtfSpecialSeedGenerator",
    ErrorRescaling = cms.double(50.0),
    OrderedHitsFactoryPSets = cms.VPSet(cms.PSet(
        ComponentName = cms.string('GenericTripletGenerator'),
        LayerSrc = cms.InputTag("combinatorialcosmicseedingtripletsTOB"),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('outsideIn')
    ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECpos"),
            PropagationDirection = cms.string('alongMomentum'),
            NavigationDirection = cms.string('outsideIn')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericTripletGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingtripletsTIB"),
            PropagationDirection = cms.string('oppositeToMomentum'),
            NavigationDirection = cms.string('insideOut')
        )),
    UpperScintillatorParameters = cms.PSet(
        GlobalX = cms.double(0.0),
        GlobalY = cms.double(300.0),
        GlobalZ = cms.double(50.0),
        WidthInX = cms.double(100.0),
        LenghtInZ = cms.double(100.0)
    ),
    Charges = cms.vint32(-1),
    PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
    MaxNumberOfCosmicClusters = cms.uint32(300),
    UseScintillatorsConstraint = cms.bool(True),
    SetMomentum = cms.bool(True),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(21.2),
            originZPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.9),
            originXPos = cms.double(0.0),
            originRadius = cms.double(0.2)
        ),
        ComponentName = cms.string('GlobalRegionProducer')
    ),
    SeedsFromNegativeY = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    doClusterCheck = cms.bool(True),
    SeedsFromPositiveY = cms.bool(True),
    MaxNumberOfPixelClusters = cms.uint32(300),
    SeedMomentum = cms.double(5.0),
    LowerScintillatorParameters = cms.PSet(
        GlobalX = cms.double(0.0),
        GlobalY = cms.double(-100.0),
        GlobalZ = cms.double(50.0),
        WidthInX = cms.double(100.0),
        LenghtInZ = cms.double(100.0)
    ),
    maxSeeds = cms.int32(10000),
    CheckHitsAreOnDifferentLayers = cms.bool(False),
    ClusterCollectionLabel = cms.InputTag("siStripClusters"),
    requireBOFF = cms.bool(False)
)


process.combinatorialcosmicseedfinderP5 = cms.EDProducer("CtfSpecialSeedGenerator",
    ErrorRescaling = cms.double(50.0),
    OrderedHitsFactoryPSets = cms.VPSet(cms.PSet(
        ComponentName = cms.string('GenericTripletGenerator'),
        LayerSrc = cms.InputTag("combinatorialcosmicseedingtripletsP5"),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('outsideIn')
    ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTOBP5"),
            PropagationDirection = cms.string('alongMomentum'),
            NavigationDirection = cms.string('outsideIn')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECposP5"),
            PropagationDirection = cms.string('alongMomentum'),
            NavigationDirection = cms.string('outsideIn')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECposP5"),
            PropagationDirection = cms.string('alongMomentum'),
            NavigationDirection = cms.string('insideOut')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECnegP5"),
            PropagationDirection = cms.string('alongMomentum'),
            NavigationDirection = cms.string('outsideIn')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECnegP5"),
            PropagationDirection = cms.string('alongMomentum'),
            NavigationDirection = cms.string('insideOut')
        )),
    UpperScintillatorParameters = cms.PSet(
        GlobalX = cms.double(0.0),
        GlobalY = cms.double(300.0),
        GlobalZ = cms.double(50.0),
        WidthInX = cms.double(100.0),
        LenghtInZ = cms.double(100.0)
    ),
    Charges = cms.vint32(-1),
    PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
    MaxNumberOfCosmicClusters = cms.uint32(300),
    UseScintillatorsConstraint = cms.bool(False),
    SetMomentum = cms.bool(True),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(21.2),
            originZPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.9),
            originXPos = cms.double(0.0),
            originRadius = cms.double(0.2)
        ),
        ComponentName = cms.string('GlobalRegionProducer')
    ),
    SeedsFromNegativeY = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    doClusterCheck = cms.bool(True),
    SeedsFromPositiveY = cms.bool(True),
    MaxNumberOfPixelClusters = cms.uint32(300),
    SeedMomentum = cms.double(5.0),
    LowerScintillatorParameters = cms.PSet(
        GlobalX = cms.double(0.0),
        GlobalY = cms.double(-100.0),
        GlobalZ = cms.double(50.0),
        WidthInX = cms.double(100.0),
        LenghtInZ = cms.double(100.0)
    ),
    maxSeeds = cms.int32(10000),
    CheckHitsAreOnDifferentLayers = cms.bool(False),
    ClusterCollectionLabel = cms.InputTag("siStripClusters"),
    requireBOFF = cms.bool(True)
)


process.combinatorialcosmicseedfinderP5Bottom = cms.EDProducer("CtfSpecialSeedGenerator",
    ErrorRescaling = cms.double(50.0),
    OrderedHitsFactoryPSets = cms.VPSet(cms.PSet(
        ComponentName = cms.string('GenericTripletGenerator'),
        LayerSrc = cms.InputTag("combinatorialcosmicseedingtripletsP5Bottom"),
        PropagationDirection = cms.string('oppositeToMomentum'),
        NavigationDirection = cms.string('outsideIn')
    ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTOBP5Bottom"),
            PropagationDirection = cms.string('oppositeToMomentum'),
            NavigationDirection = cms.string('outsideIn')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECposP5Bottom"),
            PropagationDirection = cms.string('oppositeToMomentum'),
            NavigationDirection = cms.string('outsideIn')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECposP5Bottom"),
            PropagationDirection = cms.string('oppositeToMomentum'),
            NavigationDirection = cms.string('insideOut')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECnegP5Bottom"),
            PropagationDirection = cms.string('oppositeToMomentum'),
            NavigationDirection = cms.string('outsideIn')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECnegP5Bottom"),
            PropagationDirection = cms.string('oppositeToMomentum'),
            NavigationDirection = cms.string('insideOut')
        )),
    UpperScintillatorParameters = cms.PSet(
        GlobalX = cms.double(0.0),
        GlobalY = cms.double(300.0),
        GlobalZ = cms.double(50.0),
        WidthInX = cms.double(100.0),
        LenghtInZ = cms.double(100.0)
    ),
    Charges = cms.vint32(-1),
    PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
    MaxNumberOfCosmicClusters = cms.uint32(150),
    UseScintillatorsConstraint = cms.bool(False),
    SetMomentum = cms.bool(True),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(21.2),
            originZPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.9),
            originXPos = cms.double(0.0),
            originRadius = cms.double(0.2)
        ),
        ComponentName = cms.string('GlobalRegionProducer')
    ),
    SeedsFromNegativeY = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    doClusterCheck = cms.bool(True),
    SeedsFromPositiveY = cms.bool(False),
    MaxNumberOfPixelClusters = cms.uint32(300),
    SeedMomentum = cms.double(5.0),
    LowerScintillatorParameters = cms.PSet(
        GlobalX = cms.double(0.0),
        GlobalY = cms.double(-100.0),
        GlobalZ = cms.double(50.0),
        WidthInX = cms.double(100.0),
        LenghtInZ = cms.double(100.0)
    ),
    maxSeeds = cms.int32(10000),
    CheckHitsAreOnDifferentLayers = cms.bool(False),
    ClusterCollectionLabel = cms.InputTag("siStripClustersBottom"),
    requireBOFF = cms.bool(True)
)


process.combinatorialcosmicseedfinderP5Top = cms.EDProducer("CtfSpecialSeedGenerator",
    ErrorRescaling = cms.double(50.0),
    OrderedHitsFactoryPSets = cms.VPSet(cms.PSet(
        ComponentName = cms.string('GenericTripletGenerator'),
        LayerSrc = cms.InputTag("combinatorialcosmicseedingtripletsP5Top"),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('outsideIn')
    ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTOBP5Top"),
            PropagationDirection = cms.string('alongMomentum'),
            NavigationDirection = cms.string('outsideIn')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECposP5Top"),
            PropagationDirection = cms.string('alongMomentum'),
            NavigationDirection = cms.string('outsideIn')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECposP5Top"),
            PropagationDirection = cms.string('alongMomentum'),
            NavigationDirection = cms.string('insideOut')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECnegP5Top"),
            PropagationDirection = cms.string('alongMomentum'),
            NavigationDirection = cms.string('outsideIn')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECnegP5Top"),
            PropagationDirection = cms.string('alongMomentum'),
            NavigationDirection = cms.string('insideOut')
        )),
    UpperScintillatorParameters = cms.PSet(
        GlobalX = cms.double(0.0),
        GlobalY = cms.double(300.0),
        GlobalZ = cms.double(50.0),
        WidthInX = cms.double(100.0),
        LenghtInZ = cms.double(100.0)
    ),
    Charges = cms.vint32(-1),
    PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
    MaxNumberOfCosmicClusters = cms.uint32(150),
    UseScintillatorsConstraint = cms.bool(False),
    SetMomentum = cms.bool(True),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(21.2),
            originZPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.9),
            originXPos = cms.double(0.0),
            originRadius = cms.double(0.2)
        ),
        ComponentName = cms.string('GlobalRegionProducer')
    ),
    SeedsFromNegativeY = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    doClusterCheck = cms.bool(True),
    SeedsFromPositiveY = cms.bool(True),
    MaxNumberOfPixelClusters = cms.uint32(300),
    SeedMomentum = cms.double(5.0),
    LowerScintillatorParameters = cms.PSet(
        GlobalX = cms.double(0.0),
        GlobalY = cms.double(-100.0),
        GlobalZ = cms.double(50.0),
        WidthInX = cms.double(100.0),
        LenghtInZ = cms.double(100.0)
    ),
    maxSeeds = cms.int32(10000),
    CheckHitsAreOnDifferentLayers = cms.bool(False),
    ClusterCollectionLabel = cms.InputTag("siStripClustersTop"),
    requireBOFF = cms.bool(True)
)


process.combinatorialcosmicseedingpairsTECnegP5 = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('TEC1_neg+TEC2_neg', 
        'TEC2_neg+TEC3_neg', 
        'TEC3_neg+TEC4_neg', 
        'TEC4_neg+TEC5_neg', 
        'TEC5_neg+TEC6_neg', 
        'TEC6_neg+TEC7_neg', 
        'TEC7_neg+TEC8_neg', 
        'TEC8_neg+TEC9_neg'),
    TEC = cms.PSet(
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(7)
    )
)


process.combinatorialcosmicseedingpairsTECnegP5Bottom = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('TEC1_neg+TEC2_neg', 
        'TEC2_neg+TEC3_neg', 
        'TEC3_neg+TEC4_neg', 
        'TEC4_neg+TEC5_neg', 
        'TEC5_neg+TEC6_neg', 
        'TEC6_neg+TEC7_neg', 
        'TEC7_neg+TEC8_neg', 
        'TEC8_neg+TEC9_neg'),
    TEC = cms.PSet(
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit"),
        maxRing = cms.int32(7)
    )
)


process.combinatorialcosmicseedingpairsTECnegP5Top = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('TEC1_neg+TEC2_neg', 
        'TEC2_neg+TEC3_neg', 
        'TEC3_neg+TEC4_neg', 
        'TEC4_neg+TEC5_neg', 
        'TEC5_neg+TEC6_neg', 
        'TEC6_neg+TEC7_neg', 
        'TEC7_neg+TEC8_neg', 
        'TEC8_neg+TEC9_neg'),
    TEC = cms.PSet(
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit"),
        maxRing = cms.int32(7)
    )
)


process.combinatorialcosmicseedingpairsTECposP5 = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('TEC1_pos+TEC2_pos', 
        'TEC2_pos+TEC3_pos', 
        'TEC3_pos+TEC4_pos', 
        'TEC4_pos+TEC5_pos', 
        'TEC5_pos+TEC6_pos', 
        'TEC6_pos+TEC7_pos', 
        'TEC7_pos+TEC8_pos', 
        'TEC8_pos+TEC9_pos'),
    TEC = cms.PSet(
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(7)
    )
)


process.combinatorialcosmicseedingpairsTECposP5Bottom = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('TEC1_pos+TEC2_pos', 
        'TEC2_pos+TEC3_pos', 
        'TEC3_pos+TEC4_pos', 
        'TEC4_pos+TEC5_pos', 
        'TEC5_pos+TEC6_pos', 
        'TEC6_pos+TEC7_pos', 
        'TEC7_pos+TEC8_pos', 
        'TEC8_pos+TEC9_pos'),
    TEC = cms.PSet(
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit"),
        maxRing = cms.int32(7)
    )
)


process.combinatorialcosmicseedingpairsTECposP5Top = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('TEC1_pos+TEC2_pos', 
        'TEC2_pos+TEC3_pos', 
        'TEC3_pos+TEC4_pos', 
        'TEC4_pos+TEC5_pos', 
        'TEC5_pos+TEC6_pos', 
        'TEC6_pos+TEC7_pos', 
        'TEC7_pos+TEC8_pos', 
        'TEC8_pos+TEC9_pos'),
    TEC = cms.PSet(
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit"),
        maxRing = cms.int32(7)
    )
)


process.combinatorialcosmicseedingpairsTOBP5 = cms.EDProducer("SeedingLayersEDProducer",
    TOB5 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB4 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB6 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(True),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(7)
    ),
    TIB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TIB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    layerList = cms.vstring('TOB5+TOB6', 
        'TOB4+TOB5')
)


process.combinatorialcosmicseedingpairsTOBP5Bottom = cms.EDProducer("SeedingLayersEDProducer",
    TOB5 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    TOB4 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB6 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    TOB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    TOB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(True),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit"),
        maxRing = cms.int32(7)
    ),
    TIB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TIB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    layerList = cms.vstring('TOB5+TOB6', 
        'TOB4+TOB5')
)


process.combinatorialcosmicseedingpairsTOBP5Top = cms.EDProducer("SeedingLayersEDProducer",
    TOB5 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    TOB4 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB6 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    TOB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    TOB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(True),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit"),
        maxRing = cms.int32(7)
    ),
    TIB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TIB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    layerList = cms.vstring('TOB5+TOB6', 
        'TOB4+TOB5')
)


process.combinatorialcosmicseedingtripletsP5 = cms.EDProducer("SeedingLayersEDProducer",
    TOB5 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB4 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB6 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(True),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(7)
    ),
    TIB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TIB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    layerList = cms.vstring('TOB4+TOB5+TOB6', 
        'TOB3+TOB5+TOB6', 
        'TOB3+TOB4+TOB5', 
        'TOB2+TOB4+TOB5', 
        'TOB3+TOB4+TOB6', 
        'TOB2+TOB4+TOB6')
)


process.combinatorialcosmicseedingtripletsP5Bottom = cms.EDProducer("SeedingLayersEDProducer",
    TOB5 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    TOB4 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB6 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    TOB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    TOB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(True),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit"),
        maxRing = cms.int32(7)
    ),
    TIB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TIB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    layerList = cms.vstring('TOB4+TOB5+TOB6', 
        'TOB3+TOB5+TOB6', 
        'TOB3+TOB4+TOB5', 
        'TOB2+TOB4+TOB5', 
        'TOB3+TOB4+TOB6', 
        'TOB2+TOB4+TOB6')
)


process.combinatorialcosmicseedingtripletsP5Top = cms.EDProducer("SeedingLayersEDProducer",
    TOB5 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    TOB4 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB6 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    TOB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    TOB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(True),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit"),
        maxRing = cms.int32(7)
    ),
    TIB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TIB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    layerList = cms.vstring('TOB4+TOB5+TOB6', 
        'TOB3+TOB5+TOB6', 
        'TOB3+TOB4+TOB5', 
        'TOB2+TOB4+TOB5', 
        'TOB3+TOB4+TOB6', 
        'TOB2+TOB4+TOB6')
)


process.combinedP5SeedsForCTF = cms.EDProducer("SeedCombiner",
    seedCollections = cms.VInputTag(cms.InputTag("combinatorialcosmicseedfinderP5"), cms.InputTag("simpleCosmicBONSeeds")),
    PairCollection = cms.InputTag("combinatorialcosmicseedfinderP5"),
    TripletCollection = cms.InputTag("simpleCosmicBONSeeds")
)


process.combinedP5SeedsForCTFBottom = cms.EDProducer("SeedCombiner",
    seedCollections = cms.VInputTag(cms.InputTag("combinatorialcosmicseedfinderP5Bottom"), cms.InputTag("simpleCosmicBONSeedsBottom"))
)


process.combinedP5SeedsForCTFTop = cms.EDProducer("SeedCombiner",
    seedCollections = cms.VInputTag(cms.InputTag("combinatorialcosmicseedfinderP5Top"), cms.InputTag("simpleCosmicBONSeedsTop"))
)


process.conv2Clusters = cms.EDProducer("TrackClusterRemover",
    trajectories = cms.InputTag("convStepTracks"),
    oldClusterRemovalInfo = cms.InputTag("convClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag("convStepSelector","convStep"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.conv2LayerPairs = cms.EDProducer("SeedingLayersEDProducer",
    TOB5 = cms.PSet(
        skipClusters = cms.InputTag("conv2Clusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB4 = cms.PSet(
        skipClusters = cms.InputTag("conv2Clusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        skipClusters = cms.InputTag("conv2Clusters"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB6 = cms.PSet(
        skipClusters = cms.InputTag("conv2Clusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB1 = cms.PSet(
        skipClusters = cms.InputTag("conv2Clusters"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TID3 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        skipClusters = cms.InputTag("conv2Clusters"),
        maxRing = cms.int32(2)
    ),
    TOB3 = cms.PSet(
        skipClusters = cms.InputTag("conv2Clusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB2 = cms.PSet(
        skipClusters = cms.InputTag("conv2Clusters"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
        skipClusters = cms.InputTag("conv2Clusters"),
        maxRing = cms.int32(7),
        minRing = cms.int32(1)
    ),
    layerList = cms.vstring('BPix1+BPix2', 
        'BPix2+BPix3', 
        'BPix2+FPix1_pos', 
        'BPix2+FPix1_neg', 
        'BPix2+FPix2_pos', 
        'BPix2+FPix2_neg', 
        'FPix1_pos+FPix2_pos', 
        'FPix1_neg+FPix2_neg', 
        'BPix3+TIB1', 
        'TIB1+TID1_pos', 
        'TIB1+TID1_neg', 
        'TIB1+TID2_pos', 
        'TIB1+TID2_neg', 
        'TIB1+TIB2', 
        'TIB2+TID1_pos', 
        'TIB2+TID1_neg', 
        'TIB2+TID2_pos', 
        'TIB2+TID2_neg', 
        'TIB2+TIB3', 
        'TIB3+TIB4', 
        'TIB3+TID1_pos', 
        'TIB3+TID1_neg', 
        'TIB4+TOB1', 
        'TOB1+TOB2', 
        'TOB1+TEC1_pos', 
        'TOB1+TEC1_neg', 
        'TOB2+TOB3', 
        'TOB2+TEC1_pos', 
        'TOB2+TEC1_neg', 
        'TOB3+TOB4', 
        'TOB3+TEC1_pos', 
        'TOB3+TEC1_neg', 
        'TOB4+TOB5', 
        'TOB5+TOB6', 
        'TID1_pos+TID2_pos', 
        'TID2_pos+TID3_pos', 
        'TID3_pos+TEC1_pos', 
        'TID1_neg+TID2_neg', 
        'TID2_neg+TID3_neg', 
        'TID3_neg+TEC1_neg', 
        'TEC1_pos+TEC2_pos', 
        'TEC2_pos+TEC3_pos', 
        'TEC3_pos+TEC4_pos', 
        'TEC4_pos+TEC5_pos', 
        'TEC5_pos+TEC6_pos', 
        'TEC6_pos+TEC7_pos', 
        'TEC7_pos+TEC8_pos', 
        'TEC1_neg+TEC2_neg', 
        'TEC2_neg+TEC3_neg', 
        'TEC3_neg+TEC4_neg', 
        'TEC4_neg+TEC5_neg', 
        'TEC5_neg+TEC6_neg', 
        'TEC6_neg+TEC7_neg', 
        'TEC7_neg+TEC8_neg'),
    TID2 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        skipClusters = cms.InputTag("conv2Clusters"),
        maxRing = cms.int32(2)
    ),
    FPix = cms.PSet(
        skipClusters = cms.InputTag("conv2Clusters"),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    TIB2 = cms.PSet(
        skipClusters = cms.InputTag("conv2Clusters"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TIB4 = cms.PSet(
        skipClusters = cms.InputTag("conv2Clusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TID1 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        skipClusters = cms.InputTag("conv2Clusters"),
        maxRing = cms.int32(2)
    ),
    BPix = cms.PSet(
        skipClusters = cms.InputTag("conv2Clusters"),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    TIB3 = cms.PSet(
        skipClusters = cms.InputTag("conv2Clusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    )
)


process.conv2StepSelector = cms.EDProducer("MultiTrackSelector",
    src = cms.InputTag("conv2StepTracks"),
    trackSelectors = cms.VPSet(cms.PSet(
        max_d0 = cms.double(100.0),
        minNumber3DLayers = cms.uint32(1),
        max_lostHitFraction = cms.double(1.0),
        applyAbsCutsIfNoPV = cms.bool(False),
        qualityBit = cms.string('loose'),
        minNumberLayers = cms.uint32(3),
        chi2n_par = cms.double(3.0),
        nSigmaZ = cms.double(4.0),
        dz_par2 = cms.vdouble(5.0, 8.0),
        applyAdaptedPVCuts = cms.bool(False),
        min_eta = cms.double(-9999.0),
        dz_par1 = cms.vdouble(5.0, 8.0),
        copyTrajectories = cms.untracked.bool(False),
        vtxNumber = cms.int32(-1),
        keepAllTracks = cms.bool(False),
        maxNumberLostLayers = cms.uint32(1),
        max_relpterr = cms.double(9999.0),
        copyExtras = cms.untracked.bool(True),
        vertexCut = cms.string('ndof>=2&!isFake'),
        max_z0 = cms.double(100.0),
        min_nhits = cms.uint32(0),
        name = cms.string('conv2StepLoose'),
        max_minMissHitOutOrIn = cms.int32(99),
        chi2n_no1Dmod_par = cms.double(9999),
        res_par = cms.vdouble(0.003, 0.001),
        max_eta = cms.double(9999.0),
        d0_par2 = cms.vdouble(5.0, 8.0),
        d0_par1 = cms.vdouble(5.0, 8.0),
        preFilterName = cms.string(''),
        minHitsToBypassChecks = cms.uint32(20)
    ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(1),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(3),
            chi2n_par = cms.double(2.5),
            dz_par1 = cms.vdouble(5.0, 8.0),
            dz_par2 = cms.vdouble(5.0, 8.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(1),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('conv2StepTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('conv2StepLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(5.0, 8.0),
            d0_par1 = cms.vdouble(5.0, 8.0),
            res_par = cms.vdouble(0.003, 0.001),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(1),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(3),
            chi2n_par = cms.double(2.0),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(5.0, 8.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(5.0, 8.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(1),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('conv2Step'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.001),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(5.0, 8.0),
            d0_par1 = cms.vdouble(5.0, 8.0),
            preFilterName = cms.string('conv2StepTight'),
            minHitsToBypassChecks = cms.uint32(20)
        )),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vertices = cms.InputTag("pixelVertices"),
    useVtxError = cms.bool(False),
    useVertices = cms.bool(True)
)


process.conv2StepTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("conv2TrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('conv2StepFitterSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('iter9'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.conv2TrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("photonConvTrajSeedFromQuadruplets","conv2SeedCandidates"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('conv2CkfTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder')
)


process.convClusters = cms.EDProducer("TrackClusterRemover",
    trajectories = cms.InputTag("tobTecStepTracks"),
    oldClusterRemovalInfo = cms.InputTag("tobTecStepClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag("tobTecStepSelector","tobTecStep"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.convLayerPairs = cms.EDProducer("SeedingLayersEDProducer",
    TOB5 = cms.PSet(
        skipClusters = cms.InputTag("convClusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB4 = cms.PSet(
        skipClusters = cms.InputTag("convClusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        skipClusters = cms.InputTag("convClusters"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB6 = cms.PSet(
        skipClusters = cms.InputTag("convClusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB1 = cms.PSet(
        skipClusters = cms.InputTag("convClusters"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TID3 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        skipClusters = cms.InputTag("convClusters"),
        maxRing = cms.int32(2)
    ),
    TOB3 = cms.PSet(
        skipClusters = cms.InputTag("convClusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB2 = cms.PSet(
        skipClusters = cms.InputTag("convClusters"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
        skipClusters = cms.InputTag("convClusters"),
        maxRing = cms.int32(7),
        minRing = cms.int32(1)
    ),
    layerList = cms.vstring('BPix1+BPix2', 
        'BPix2+BPix3', 
        'BPix2+FPix1_pos', 
        'BPix2+FPix1_neg', 
        'BPix2+FPix2_pos', 
        'BPix2+FPix2_neg', 
        'FPix1_pos+FPix2_pos', 
        'FPix1_neg+FPix2_neg', 
        'BPix3+TIB1', 
        'BPix3+TIB2', 
        'TIB1+TID1_pos', 
        'TIB1+TID1_neg', 
        'TIB1+TID2_pos', 
        'TIB1+TID2_neg', 
        'TIB1+TIB2', 
        'TIB1+TIB3', 
        'TIB2+TID1_pos', 
        'TIB2+TID1_neg', 
        'TIB2+TID2_pos', 
        'TIB2+TID2_neg', 
        'TIB2+TIB3', 
        'TIB2+TIB4', 
        'TIB3+TIB4', 
        'TIB3+TOB1', 
        'TIB3+TID1_pos', 
        'TIB3+TID1_neg', 
        'TIB4+TOB1', 
        'TIB4+TOB2', 
        'TOB1+TOB2', 
        'TOB1+TOB3', 
        'TOB1+TEC1_pos', 
        'TOB1+TEC1_neg', 
        'TOB2+TOB3', 
        'TOB2+TOB4', 
        'TOB2+TEC1_pos', 
        'TOB2+TEC1_neg', 
        'TID1_pos+TID2_pos', 
        'TID2_pos+TID3_pos', 
        'TID3_pos+TEC1_pos', 
        'TID1_neg+TID2_neg', 
        'TID2_neg+TID3_neg', 
        'TID3_neg+TEC1_neg', 
        'TEC1_pos+TEC2_pos', 
        'TEC2_pos+TEC3_pos', 
        'TEC3_pos+TEC4_pos', 
        'TEC4_pos+TEC5_pos', 
        'TEC5_pos+TEC6_pos', 
        'TEC6_pos+TEC7_pos', 
        'TEC7_pos+TEC8_pos', 
        'TEC1_neg+TEC2_neg', 
        'TEC2_neg+TEC3_neg', 
        'TEC3_neg+TEC4_neg', 
        'TEC4_neg+TEC5_neg', 
        'TEC5_neg+TEC6_neg', 
        'TEC6_neg+TEC7_neg', 
        'TEC7_neg+TEC8_neg'),
    TID2 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        skipClusters = cms.InputTag("convClusters"),
        maxRing = cms.int32(2)
    ),
    FPix = cms.PSet(
        skipClusters = cms.InputTag("convClusters"),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    TIB2 = cms.PSet(
        skipClusters = cms.InputTag("convClusters"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TIB4 = cms.PSet(
        skipClusters = cms.InputTag("convClusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TID1 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        skipClusters = cms.InputTag("convClusters"),
        maxRing = cms.int32(2)
    ),
    BPix = cms.PSet(
        skipClusters = cms.InputTag("convClusters"),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    TIB3 = cms.PSet(
        skipClusters = cms.InputTag("convClusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    )
)


process.convStepSelector = cms.EDProducer("MultiTrackSelector",
    src = cms.InputTag("convStepTracks"),
    trackSelectors = cms.VPSet(cms.PSet(
        max_d0 = cms.double(100.0),
        minNumber3DLayers = cms.uint32(1),
        max_lostHitFraction = cms.double(1.0),
        applyAbsCutsIfNoPV = cms.bool(False),
        qualityBit = cms.string('loose'),
        minNumberLayers = cms.uint32(3),
        chi2n_par = cms.double(3.0),
        nSigmaZ = cms.double(4.0),
        dz_par2 = cms.vdouble(5.0, 8.0),
        applyAdaptedPVCuts = cms.bool(False),
        min_eta = cms.double(-9999.0),
        dz_par1 = cms.vdouble(5.0, 8.0),
        copyTrajectories = cms.untracked.bool(False),
        vtxNumber = cms.int32(-1),
        keepAllTracks = cms.bool(False),
        maxNumberLostLayers = cms.uint32(1),
        max_relpterr = cms.double(9999.0),
        copyExtras = cms.untracked.bool(True),
        vertexCut = cms.string('ndof>=2&!isFake'),
        max_z0 = cms.double(100.0),
        min_nhits = cms.uint32(0),
        name = cms.string('convStepLoose'),
        max_minMissHitOutOrIn = cms.int32(99),
        chi2n_no1Dmod_par = cms.double(9999),
        res_par = cms.vdouble(0.003, 0.001),
        max_eta = cms.double(9999.0),
        d0_par2 = cms.vdouble(5.0, 8.0),
        d0_par1 = cms.vdouble(5.0, 8.0),
        preFilterName = cms.string(''),
        minHitsToBypassChecks = cms.uint32(20)
    ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(1),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(3),
            chi2n_par = cms.double(2.5),
            dz_par1 = cms.vdouble(5.0, 8.0),
            dz_par2 = cms.vdouble(5.0, 8.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(1),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('convStepTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('convStepLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(5.0, 8.0),
            d0_par1 = cms.vdouble(5.0, 8.0),
            res_par = cms.vdouble(0.003, 0.001),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(1),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(3),
            chi2n_par = cms.double(2.0),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(5.0, 8.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(5.0, 8.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(1),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('convStep'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.001),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(5.0, 8.0),
            d0_par1 = cms.vdouble(5.0, 8.0),
            preFilterName = cms.string('convStepTight'),
            minHitsToBypassChecks = cms.uint32(20)
        )),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vertices = cms.InputTag("pixelVertices"),
    useVtxError = cms.bool(False),
    useVertices = cms.bool(True)
)


process.convStepTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("convTrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('convStepFitterSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('iter8'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.convTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("photonConvTrajSeedFromSingleLeg","convSeedCandidates"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('convCkfTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    clustersToSkip = cms.InputTag("convClusters")
)


process.conversionStepTracks = cms.EDProducer("TrackListMerger",
    ShareFrac = cms.double(0.19),
    writeOnlyTrkQuals = cms.bool(False),
    MinPT = cms.double(0.05),
    allowFirstHitShare = cms.bool(True),
    copyExtras = cms.untracked.bool(True),
    Epsilon = cms.double(-0.001),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("convStepSelector","convStep")),
    indivShareFrac = cms.vdouble(1.0, 1.0),
    makeReKeyedSeeds = cms.untracked.bool(False),
    MaxNormalizedChisq = cms.double(1000.0),
    FoundHitBonus = cms.double(5.0),
    setsToMerge = cms.VPSet(cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(1)
    )),
    MinFound = cms.int32(3),
    hasSelector = cms.vint32(1),
    TrackProducers = cms.VInputTag(cms.InputTag("convStepTracks")),
    LostHitPenalty = cms.double(20.0),
    newQuality = cms.string('confirmed')
)


process.cosmicCandidateFinder = cms.EDProducer("CosmicTrackFinder",
    MinHits = cms.int32(4),
    HitProducer = cms.string('siStripRecHits'),
    pixelRecHits = cms.InputTag("siPixelRecHits"),
    useHitsSplitting = cms.bool(True),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    Chi2Cut = cms.double(30.0),
    TTRHBuilder = cms.string('WithTrackAngle'),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    debug = cms.untracked.bool(True),
    GeometricStructure = cms.untracked.string('MTCC'),
    cosmicSeeds = cms.InputTag("cosmicseedfinder")
)


process.cosmicCandidateFinderP5 = cms.EDProducer("CosmicTrackFinder",
    MinHits = cms.int32(4),
    HitProducer = cms.string('siStripRecHits'),
    pixelRecHits = cms.InputTag("siPixelRecHits"),
    useHitsSplitting = cms.bool(True),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    Chi2Cut = cms.double(30.0),
    TTRHBuilder = cms.string('WithTrackAngle'),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    debug = cms.untracked.bool(True),
    GeometricStructure = cms.untracked.string('STANDARD'),
    cosmicSeeds = cms.InputTag("cosmicseedfinderP5")
)


process.cosmicCandidateFinderP5Bottom = cms.EDProducer("CosmicTrackFinder",
    MinHits = cms.int32(4),
    HitProducer = cms.string('siStripRecHitsBottom'),
    pixelRecHits = cms.InputTag("siPixelRecHitsBottom"),
    useHitsSplitting = cms.bool(True),
    matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
    stereorecHits = cms.InputTag("siStripMatchedRecHitsBottom","stereoRecHit"),
    Chi2Cut = cms.double(30.0),
    TTRHBuilder = cms.string('WithTrackAngle'),
    rphirecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit"),
    debug = cms.untracked.bool(True),
    GeometricStructure = cms.untracked.string('STANDARD'),
    cosmicSeeds = cms.InputTag("cosmicseedfinderP5Bottom")
)


process.cosmicCandidateFinderP5Top = cms.EDProducer("CosmicTrackFinder",
    MinHits = cms.int32(4),
    HitProducer = cms.string('siStripRecHitsTop'),
    pixelRecHits = cms.InputTag("siPixelRecHitsTop"),
    useHitsSplitting = cms.bool(True),
    matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
    stereorecHits = cms.InputTag("siStripMatchedRecHitsTop","stereoRecHit"),
    Chi2Cut = cms.double(30.0),
    TTRHBuilder = cms.string('WithTrackAngle'),
    rphirecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit"),
    debug = cms.untracked.bool(True),
    GeometricStructure = cms.untracked.string('STANDARD'),
    cosmicSeeds = cms.InputTag("cosmicseedfinderP5Top")
)


process.cosmicTrackSplitter = cms.EDProducer("CosmicTrackSplitter",
    dxyCut = cms.double(9999.0),
    stripBackInvalidHits = cms.bool(True),
    stripAllInvalidHits = cms.bool(False),
    tracks = cms.InputTag("cosmictrackfinderCosmics"),
    excludePixelHits = cms.bool(False),
    tjTkAssociationMapTag = cms.InputTag("cosmictrackfinderCosmics"),
    replaceWithInactiveHits = cms.bool(False),
    dzCut = cms.double(9999.0),
    minimumHits = cms.uint32(6),
    detsToIgnore = cms.vuint32(),
    stripFrontInvalidHits = cms.bool(True)
)


process.cosmicseedfinder = cms.EDProducer("CosmicSeedGenerator",
    originRadius = cms.double(150.0),
    originHalfLength = cms.double(90.0),
    originZPosition = cms.double(0.0),
    GeometricStructure = cms.untracked.string('STANDARD'),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
    MaxNumberOfCosmicClusters = cms.uint32(300),
    SeedPt = cms.double(5.0),
    HitsForSeeds = cms.untracked.string('pairs'),
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    NegativeYOnly = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    PositiveYOnly = cms.bool(False),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    MaxNumberOfPixelClusters = cms.uint32(300),
    doClusterCheck = cms.bool(True),
    maxSeeds = cms.int32(10000),
    ClusterCollectionLabel = cms.InputTag("siStripClusters"),
    ptMin = cms.double(0.9)
)


process.cosmicseedfinderP5 = cms.EDProducer("CosmicSeedGenerator",
    originRadius = cms.double(150.0),
    originHalfLength = cms.double(90.0),
    originZPosition = cms.double(0.0),
    GeometricStructure = cms.untracked.string('STANDARD'),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
    MaxNumberOfCosmicClusters = cms.uint32(300),
    SeedPt = cms.double(5.0),
    HitsForSeeds = cms.untracked.string('pairs'),
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    NegativeYOnly = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    PositiveYOnly = cms.bool(False),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    MaxNumberOfPixelClusters = cms.uint32(300),
    doClusterCheck = cms.bool(True),
    maxSeeds = cms.int32(10000),
    ClusterCollectionLabel = cms.InputTag("siStripClusters"),
    ptMin = cms.double(0.9)
)


process.cosmicseedfinderP5Bottom = cms.EDProducer("CosmicSeedGenerator",
    originRadius = cms.double(150.0),
    originHalfLength = cms.double(90.0),
    originZPosition = cms.double(0.0),
    GeometricStructure = cms.untracked.string('STANDARD'),
    matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
    PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
    MaxNumberOfCosmicClusters = cms.uint32(150),
    SeedPt = cms.double(5.0),
    HitsForSeeds = cms.untracked.string('pairs'),
    stereorecHits = cms.InputTag("siStripMatchedRecHitsBottom","stereoRecHit"),
    NegativeYOnly = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    PositiveYOnly = cms.bool(False),
    rphirecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit"),
    MaxNumberOfPixelClusters = cms.uint32(300),
    doClusterCheck = cms.bool(True),
    maxSeeds = cms.int32(10000),
    ClusterCollectionLabel = cms.InputTag("siStripClustersBottom"),
    ptMin = cms.double(0.9)
)


process.cosmicseedfinderP5Top = cms.EDProducer("CosmicSeedGenerator",
    originRadius = cms.double(150.0),
    originHalfLength = cms.double(90.0),
    originZPosition = cms.double(0.0),
    GeometricStructure = cms.untracked.string('STANDARD'),
    matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
    PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
    MaxNumberOfCosmicClusters = cms.uint32(150),
    SeedPt = cms.double(5.0),
    HitsForSeeds = cms.untracked.string('pairs'),
    stereorecHits = cms.InputTag("siStripMatchedRecHitsTop","stereoRecHit"),
    NegativeYOnly = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    PositiveYOnly = cms.bool(True),
    rphirecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit"),
    MaxNumberOfPixelClusters = cms.uint32(300),
    doClusterCheck = cms.bool(True),
    maxSeeds = cms.int32(10000),
    ClusterCollectionLabel = cms.InputTag("siStripClustersTop"),
    ptMin = cms.double(0.9)
)


process.cosmictrackfinderCosmics = cms.EDProducer("TrackProducer",
    src = cms.InputTag("cosmicCandidateFinderP5"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('RKFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('cosmic'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.cosmictrackfinderP5 = cms.EDProducer("CosmicTrackSelector",
    keepAllTracks = cms.bool(False),
    maxNumberLostLayers = cms.uint32(999),
    max_d0 = cms.double(110.0),
    minNumber3DLayers = cms.uint32(0),
    src = cms.InputTag("cosmictrackfinderCosmics"),
    copyExtras = cms.untracked.bool(True),
    min_pt = cms.double(1.0),
    copyTrajectories = cms.untracked.bool(True),
    qualityBit = cms.string(''),
    minNumberLayers = cms.uint32(0),
    chi2n_par = cms.double(10.0),
    max_eta = cms.double(2.0),
    min_nPixelHit = cms.uint32(0),
    min_nHit = cms.uint32(5),
    max_z0 = cms.double(300.0),
    beamspot = cms.InputTag("offlineBeamSpot")
)


process.cosmictrackfinderP5Bottom = cms.EDProducer("TrackProducer",
    src = cms.InputTag("cosmicCandidateFinderP5Bottom"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag("topBottomClusterInfoProducerBottom"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('RKFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('cosmic'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.cosmictrackfinderP5Top = cms.EDProducer("TrackProducer",
    src = cms.InputTag("cosmicCandidateFinderP5Top"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag("topBottomClusterInfoProducerTop"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('RKFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('cosmic'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.ctfCombinedSeeds = cms.EDProducer("TrackProducer",
    src = cms.InputTag("ckfTrackCandidatesCombinedSeeds"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('undefAlgorithm'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.ctfNoOverlaps = cms.EDProducer("TrackProducer",
    src = cms.InputTag("ckfTrackCandidatesNoOverlaps"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('undefAlgorithm'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.ctfPixelLess = cms.EDProducer("TrackProducer",
    src = cms.InputTag("ckfTrackCandidatesPixelLess"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('RKFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('undefAlgorithm'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.ctfWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("ckfTrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('undefAlgorithm'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.ctfWithMaterialTracksCosmics = cms.EDProducer("TrackProducer",
    src = cms.InputTag("ckfTrackCandidatesP5"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('FittingSmootherRKP5'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('ctf'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(True),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.ctfWithMaterialTracksP5 = cms.EDProducer("CosmicTrackSelector",
    keepAllTracks = cms.bool(False),
    maxNumberLostLayers = cms.uint32(999),
    max_d0 = cms.double(110.0),
    minNumber3DLayers = cms.uint32(0),
    src = cms.InputTag("ctfWithMaterialTracksCosmics"),
    copyExtras = cms.untracked.bool(True),
    min_pt = cms.double(1.0),
    copyTrajectories = cms.untracked.bool(True),
    qualityBit = cms.string(''),
    minNumberLayers = cms.uint32(0),
    chi2n_par = cms.double(10.0),
    max_eta = cms.double(2.0),
    min_nPixelHit = cms.uint32(0),
    min_nHit = cms.uint32(5),
    max_z0 = cms.double(300.0),
    beamspot = cms.InputTag("offlineBeamSpot")
)


process.ctfWithMaterialTracksP5Bottom = cms.EDProducer("TrackProducer",
    src = cms.InputTag("ckfTrackCandidatesP5Bottom"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag("topBottomClusterInfoProducerBottom"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('FittingSmootherRKP5'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('ctf'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(True),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.ctfWithMaterialTracksP5LHCNavigation = cms.EDProducer("TrackProducer",
    src = cms.InputTag("ckfTrackCandidatesP5LHCNavigation"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('FittingSmootherRKP5'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    GeometricInnerState = cms.bool(True),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    AlgorithmName = cms.string('ctf'),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.ctfWithMaterialTracksP5Top = cms.EDProducer("TrackProducer",
    src = cms.InputTag("ckfTrackCandidatesP5Top"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag("topBottomClusterInfoProducerTop"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('FittingSmootherRKP5'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('ctf'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(True),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.dedxDiscrimASmi = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("generalTracks"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('asmirnovDiscrim'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0),
    trajectoryTrackAssociation = cms.InputTag("generalTracks")
)


process.dedxDiscrimASmiCTF = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    trajectoryTrackAssociation = cms.InputTag("ctfWithMaterialTracksP5"),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("ctfWithMaterialTracksP5"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('asmirnovDiscrim'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0)
)


process.dedxDiscrimASmiCTFP5LHC = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    trajectoryTrackAssociation = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation"),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('asmirnovDiscrim'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0)
)


process.dedxDiscrimASmiCosmicTF = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    trajectoryTrackAssociation = cms.InputTag("cosmictrackfinderP5"),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("cosmictrackfinderP5"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('asmirnovDiscrim'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0)
)


process.dedxDiscrimASmiRS = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    trajectoryTrackAssociation = cms.InputTag("rsWithMaterialTracksP5"),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("rsWithMaterialTracksP5"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('asmirnovDiscrim'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0)
)


process.dedxDiscrimBTag = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("generalTracks"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('btagDiscrim'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0),
    trajectoryTrackAssociation = cms.InputTag("generalTracks")
)


process.dedxDiscrimProd = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("generalTracks"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('productDiscrim'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0),
    trajectoryTrackAssociation = cms.InputTag("generalTracks")
)


process.dedxDiscrimSmi = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("generalTracks"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('smirnovDiscrim'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0),
    trajectoryTrackAssociation = cms.InputTag("generalTracks")
)


process.dedxHarmonic2 = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    trajectoryTrackAssociation = cms.InputTag("generalTracks"),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("generalTracks"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('generic'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0)
)


process.dedxHarmonic2CTF = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("ctfWithMaterialTracksP5"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('generic'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0),
    trajectoryTrackAssociation = cms.InputTag("ctfWithMaterialTracksP5")
)


process.dedxHarmonic2CTFP5LHC = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('generic'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0),
    trajectoryTrackAssociation = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")
)


process.dedxHarmonic2CosmicTF = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("cosmictrackfinderP5"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('generic'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0),
    trajectoryTrackAssociation = cms.InputTag("cosmictrackfinderP5")
)


process.dedxHarmonic2RS = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("rsWithMaterialTracksP5"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('generic'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0),
    trajectoryTrackAssociation = cms.InputTag("rsWithMaterialTracksP5")
)


process.dedxMedian = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("generalTracks"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('median'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0),
    trajectoryTrackAssociation = cms.InputTag("generalTracks")
)


process.dedxTruncated40 = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("generalTracks"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('truncated'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0),
    trajectoryTrackAssociation = cms.InputTag("generalTracks")
)


process.dedxTruncated40CTF = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    trajectoryTrackAssociation = cms.InputTag("ctfWithMaterialTracksP5"),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("ctfWithMaterialTracksP5"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('truncated'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0)
)


process.dedxTruncated40CTFP5LHC = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    trajectoryTrackAssociation = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation"),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('truncated'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0)
)


process.dedxTruncated40CosmicTF = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    trajectoryTrackAssociation = cms.InputTag("cosmictrackfinderP5"),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("cosmictrackfinderP5"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('truncated'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0)
)


process.dedxTruncated40RS = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    trajectoryTrackAssociation = cms.InputTag("rsWithMaterialTracksP5"),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("rsWithMaterialTracksP5"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('truncated'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0)
)


process.dedxUnbinned = cms.EDProducer("DeDxEstimatorProducer",
    UseStrip = cms.bool(True),
    MeVperADCPixel = cms.double(3.61e-06),
    UseCalibration = cms.bool(False),
    calibrationPath = cms.string(''),
    ProbabilityMode = cms.string('Accumulation'),
    tracks = cms.InputTag("generalTracks"),
    UsePixel = cms.bool(False),
    ShapeTest = cms.bool(True),
    fraction = cms.double(0.4),
    MeVperADCStrip = cms.double(0.00095665),
    UseTrajectory = cms.bool(True),
    estimator = cms.string('unbinnedFit'),
    Reccord = cms.string('SiStripDeDxMip_3D_Rcd'),
    exponent = cms.double(-2.0),
    trajectoryTrackAssociation = cms.InputTag("generalTracks")
)


process.detachedTripletStep = cms.EDProducer("TrackListMerger",
    ShareFrac = cms.double(0.19),
    writeOnlyTrkQuals = cms.bool(True),
    MinPT = cms.double(0.05),
    allowFirstHitShare = cms.bool(True),
    copyExtras = cms.untracked.bool(False),
    Epsilon = cms.double(-0.001),
    shareFrac = cms.double(0.13),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("detachedTripletStepSelector","detachedTripletStepVtx"), cms.InputTag("detachedTripletStepSelector","detachedTripletStepTrk")),
    indivShareFrac = cms.vdouble(0.13, 0.13),
    MaxNormalizedChisq = cms.double(1000.0),
    hasSelector = cms.vint32(1, 1),
    FoundHitBonus = cms.double(5.0),
    setsToMerge = cms.VPSet(cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0, 1)
    )),
    MinFound = cms.int32(3),
    TrackProducers = cms.VInputTag(cms.InputTag("detachedTripletStepTracks"), cms.InputTag("detachedTripletStepTracks")),
    LostHitPenalty = cms.double(20.0),
    newQuality = cms.string('confirmed')
)


process.detachedTripletStepClusters = cms.EDProducer("TrackClusterRemover",
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
    trajectories = cms.InputTag("initialStepTracks"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag("initialStep"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.detachedTripletStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg'),
    MTOB = cms.PSet(

    ),
    TEC = cms.PSet(

    ),
    MTID = cms.PSet(

    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag("detachedTripletStepClusters")
    ),
    MTEC = cms.PSet(

    ),
    MTIB = cms.PSet(

    ),
    TID = cms.PSet(

    ),
    TOB = cms.PSet(

    ),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag("detachedTripletStepClusters")
    ),
    TIB = cms.PSet(

    )
)


process.detachedTripletStepSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        GeneratorPSet = cms.PSet(
            useBending = cms.bool(True),
            useFixedPreFiltering = cms.bool(False),
            maxElement = cms.uint32(100000),
            ComponentName = cms.string('PixelTripletLargeTipGenerator'),
            extraHitRPhitolerance = cms.double(0.0),
            useMultScattering = cms.bool(True),
            phiPreFiltering = cms.double(0.3),
            extraHitRZtolerance = cms.double(0.0)
        ),
        SeedingLayers = cms.InputTag("detachedTripletStepSeedLayers")
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterStripHits = cms.bool(False),
        ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        FilterPixelHits = cms.bool(True),
        FilterAtHelixStage = cms.bool(False)
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            originRadius = cms.double(1.5),
            ptMin = cms.double(0.3),
            originHalfLength = cms.double(15.0)
        ),
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsTripletOnlyCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.detachedTripletStepSelector = cms.EDProducer("MultiTrackSelector",
    src = cms.InputTag("detachedTripletStepTracks"),
    trackSelectors = cms.VPSet(cms.PSet(
        max_d0 = cms.double(100.0),
        minNumber3DLayers = cms.uint32(0),
        max_lostHitFraction = cms.double(1.0),
        applyAbsCutsIfNoPV = cms.bool(False),
        qualityBit = cms.string('loose'),
        minNumberLayers = cms.uint32(0),
        chi2n_par = cms.double(9999),
        nSigmaZ = cms.double(4.0),
        dz_par2 = cms.vdouble(1.3, 3.0),
        applyAdaptedPVCuts = cms.bool(True),
        min_eta = cms.double(-9999.0),
        dz_par1 = cms.vdouble(1.2, 3.0),
        copyTrajectories = cms.untracked.bool(False),
        vtxNumber = cms.int32(-1),
        keepAllTracks = cms.bool(False),
        maxNumberLostLayers = cms.uint32(999),
        max_relpterr = cms.double(9999.0),
        copyExtras = cms.untracked.bool(True),
        minMVA = cms.double(-0.5),
        vertexCut = cms.string('ndof>=2&!isFake'),
        max_z0 = cms.double(100.0),
        min_nhits = cms.uint32(0),
        name = cms.string('detachedTripletStepVtxLoose'),
        max_minMissHitOutOrIn = cms.int32(99),
        chi2n_no1Dmod_par = cms.double(9999),
        res_par = cms.vdouble(0.003, 0.01),
        useMVA = cms.bool(True),
        max_eta = cms.double(9999.0),
        d0_par2 = cms.vdouble(1.3, 3.0),
        d0_par1 = cms.vdouble(1.2, 3.0),
        preFilterName = cms.string(''),
        minHitsToBypassChecks = cms.uint32(20)
    ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(0),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('loose'),
            minNumberLayers = cms.uint32(0),
            chi2n_par = cms.double(9999),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(1.4, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(1.4, 4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(False),
            maxNumberLostLayers = cms.uint32(999),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            minMVA = cms.double(-0.5),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('detachedTripletStepTrkLoose'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.01),
            useMVA = cms.bool(True),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.4, 4.0),
            d0_par1 = cms.vdouble(1.4, 4.0),
            preFilterName = cms.string(''),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(3),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(3),
            chi2n_par = cms.double(0.9),
            dz_par1 = cms.vdouble(1.1, 3.0),
            dz_par2 = cms.vdouble(1.2, 3.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(1),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('detachedTripletStepVtxTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('detachedTripletStepVtxLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.2, 3.0),
            d0_par1 = cms.vdouble(1.1, 3.0),
            res_par = cms.vdouble(0.003, 0.001),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(4),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(5),
            chi2n_par = cms.double(0.5),
            dz_par1 = cms.vdouble(1.1, 4.0),
            dz_par2 = cms.vdouble(1.1, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(1),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('detachedTripletStepTrkTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('detachedTripletStepTrkLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.1, 4.0),
            d0_par1 = cms.vdouble(1.1, 4.0),
            res_par = cms.vdouble(0.003, 0.001),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(0),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(0),
            chi2n_par = cms.double(9999),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(1.1, 3.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(1.0, 3.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(999),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            minMVA = cms.double(0.5),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('detachedTripletStepVtx'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.01),
            useMVA = cms.bool(True),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.1, 3.0),
            d0_par1 = cms.vdouble(1.0, 3.0),
            preFilterName = cms.string('detachedTripletStepVtxLoose'),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(0),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(0),
            chi2n_par = cms.double(9999),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(1.0, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(1.0, 4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(999),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            minMVA = cms.double(0.5),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('detachedTripletStepTrk'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.01),
            useMVA = cms.bool(True),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.0, 4.0),
            d0_par1 = cms.vdouble(1.0, 4.0),
            preFilterName = cms.string('detachedTripletStepTrkLoose'),
            minHitsToBypassChecks = cms.uint32(20)
        )),
    GBRForestLabel = cms.string('MVASelectorIter3_13TeV_v0'),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vertices = cms.InputTag("pixelVertices"),
    useVtxError = cms.bool(False),
    useAnyMVA = cms.bool(True),
    useVertices = cms.bool(True)
)


process.detachedTripletStepTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("detachedTripletStepSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('detachedTripletStepTrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    numHitsForSeedCleaner = cms.int32(50),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('detachedTripletStepTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    clustersToSkip = cms.InputTag("detachedTripletStepClusters")
)


process.detachedTripletStepTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("detachedTripletStepTrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('iter3'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.duplicateTrackCandidates = cms.EDProducer("DuplicateTrackMerger",
    forestLabel = cms.string('MVADuplicate'),
    maxDQoP = cms.double(0.25),
    minpT = cms.double(0.2),
    maxDCA = cms.double(30.0),
    maxDdxy = cms.double(10.0),
    maxDLambda = cms.double(0.3),
    source = cms.InputTag("preDuplicateMergingGeneralTracks"),
    useInnermostState = cms.bool(True),
    maxDPhi = cms.double(0.3),
    minP = cms.double(0.4),
    minBDTG = cms.double(-0.1),
    maxDdsz = cms.double(10.0),
    minDeltaR3d = cms.double(-4.0),
    ttrhBuilderName = cms.string('WithAngleAndTemplate')
)


process.duplicateTrackSelector = cms.EDProducer("MultiTrackSelector",
    src = cms.InputTag("mergedDuplicateTracks"),
    trackSelectors = cms.VPSet(cms.PSet(
        max_d0 = cms.double(100.0),
        minNumber3DLayers = cms.uint32(0),
        max_lostHitFraction = cms.double(1.0),
        applyAbsCutsIfNoPV = cms.bool(False),
        qualityBit = cms.string('loose'),
        minNumberLayers = cms.uint32(0),
        chi2n_par = cms.double(1.6),
        nSigmaZ = cms.double(4.0),
        dz_par2 = cms.vdouble(0.45, 4.0),
        applyAdaptedPVCuts = cms.bool(True),
        min_eta = cms.double(-9999.0),
        dz_par1 = cms.vdouble(0.65, 4.0),
        copyTrajectories = cms.untracked.bool(False),
        vtxNumber = cms.int32(-1),
        keepAllTracks = cms.bool(False),
        maxNumberLostLayers = cms.uint32(999),
        max_relpterr = cms.double(9999.0),
        copyExtras = cms.untracked.bool(True),
        vertexCut = cms.string('ndof>=2&!isFake'),
        max_z0 = cms.double(100.0),
        min_nhits = cms.uint32(0),
        name = cms.string('duplicateTrackSelectorLoose'),
        max_minMissHitOutOrIn = cms.int32(99),
        chi2n_no1Dmod_par = cms.double(9999),
        res_par = cms.vdouble(0.003, 0.01),
        max_eta = cms.double(9999.0),
        d0_par2 = cms.vdouble(0.55, 4.0),
        d0_par1 = cms.vdouble(0.55, 4.0),
        preFilterName = cms.string(''),
        minHitsToBypassChecks = cms.uint32(0)
    )),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vertices = cms.InputTag("pixelVertices"),
    useVtxError = cms.bool(False),
    useVertices = cms.bool(True)
)


process.earlyGeneralTracks = cms.EDProducer("TrackListMerger",
    ShareFrac = cms.double(0.19),
    writeOnlyTrkQuals = cms.bool(False),
    MinPT = cms.double(0.05),
    allowFirstHitShare = cms.bool(True),
    copyExtras = cms.untracked.bool(True),
    Epsilon = cms.double(-0.001),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStep"), cms.InputTag("jetCoreRegionalStepSelector","jetCoreRegionalStep"), cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"), cms.InputTag("pixelPairStepSelector","pixelPairStep"), cms.InputTag("detachedTripletStep"), 
        cms.InputTag("mixedTripletStep"), cms.InputTag("pixelLessStep"), cms.InputTag("tobTecStepSelector","tobTecStep")),
    indivShareFrac = cms.vdouble(1.0, 0.19, 0.16, 0.19, 0.13, 
        0.11, 0.11, 0.09),
    makeReKeyedSeeds = cms.untracked.bool(False),
    MaxNormalizedChisq = cms.double(1000.0),
    FoundHitBonus = cms.double(5.0),
    setsToMerge = cms.VPSet(cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0, 1, 2, 3, 4, 
            5, 6, 7)
    )),
    MinFound = cms.int32(3),
    hasSelector = cms.vint32(1, 1, 1, 1, 1, 
        1, 1, 1),
    TrackProducers = cms.VInputTag(cms.InputTag("initialStepTracks"), cms.InputTag("jetCoreRegionalStepTracks"), cms.InputTag("lowPtTripletStepTracks"), cms.InputTag("pixelPairStepTracks"), cms.InputTag("detachedTripletStepTracks"), 
        cms.InputTag("mixedTripletStepTracks"), cms.InputTag("pixelLessStepTracks"), cms.InputTag("tobTecStepTracks")),
    LostHitPenalty = cms.double(20.0),
    newQuality = cms.string('confirmed')
)


process.earlyMuons = cms.EDProducer("MuonIdProducer",
    TrackExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        inputTrackCollection = cms.InputTag("generalTracks"),
        BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
        ComponentName = cms.string('TrackExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        Chi2Prob_Min = cms.double(-1.0),
        DR_Veto = cms.double(0.01),
        NHits_Min = cms.uint32(0),
        Chi2Ndof_Max = cms.double(1e+64),
        Pt_Min = cms.double(-1.0),
        DepositLabel = cms.untracked.string(''),
        BeamlineOption = cms.string('BeamSpotFromEvent')
    ),
    maxAbsEta = cms.double(3.0),
    fillGlobalTrackRefits = cms.bool(False),
    arbitrationCleanerOptions = cms.PSet(
        Clustering = cms.bool(True),
        ME1a = cms.bool(True),
        ClusterDPhi = cms.double(0.6),
        OverlapDTheta = cms.double(0.02),
        Overlap = cms.bool(True),
        OverlapDPhi = cms.double(0.0786),
        ClusterDTheta = cms.double(0.02)
    ),
    globalTrackQualityInputTag = cms.InputTag("glbTrackQual"),
    addExtraSoftMuons = cms.bool(False),
    debugWithTruthMatching = cms.bool(False),
    CaloExtractorPSet = cms.PSet(
        PrintTimeReport = cms.untracked.bool(False),
        DR_Max = cms.double(1.0),
        Threshold_E = cms.double(0.2),
        DepositInstanceLabels = cms.vstring('ecal', 
            'hcal', 
            'ho'),
        Noise_HE = cms.double(0.2),
        NoiseTow_EB = cms.double(0.04),
        NoiseTow_EE = cms.double(0.15),
        Threshold_H = cms.double(0.5),
        ServiceParameters = cms.PSet(
            Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny'),
            RPCLayers = cms.bool(False),
            UseMuonNavigation = cms.untracked.bool(False)
        ),
        Noise_HO = cms.double(0.2),
        PropagatorName = cms.string('SteppingHelixPropagatorAny'),
        DepositLabel = cms.untracked.string('Cal'),
        UseRecHitsFlag = cms.bool(False),
        TrackAssociatorParameters = cms.PSet(
            muonMaxDistanceSigmaX = cms.double(0.0),
            muonMaxDistanceSigmaY = cms.double(0.0),
            CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
            dRHcal = cms.double(1.0),
            dRPreshowerPreselection = cms.double(0.2),
            CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
            useEcal = cms.bool(False),
            dREcal = cms.double(1.0),
            dREcalPreselection = cms.double(1.0),
            HORecHitCollectionLabel = cms.InputTag("horeco"),
            dRMuon = cms.double(9999.0),
            propagateAllDirections = cms.bool(True),
            muonMaxDistanceX = cms.double(5.0),
            muonMaxDistanceY = cms.double(5.0),
            useHO = cms.bool(False),
            trajectoryUncertaintyTolerance = cms.double(-1.0),
            usePreshower = cms.bool(False),
            DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
            EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
            dRHcalPreselection = cms.double(1.0),
            useMuon = cms.bool(False),
            useCalo = cms.bool(True),
            accountForTrajectoryChangeCalo = cms.bool(False),
            EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
            dRMuonPreselection = cms.double(0.2),
            truthMatch = cms.bool(False),
            HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
            useHcal = cms.bool(False)
        ),
        Threshold_HO = cms.double(0.5),
        Noise_EE = cms.double(0.1),
        Noise_EB = cms.double(0.025),
        DR_Veto_H = cms.double(0.1),
        CenterConeOnCalIntersection = cms.bool(False),
        ComponentName = cms.string('CaloExtractorByAssociator'),
        Noise_HB = cms.double(0.2),
        DR_Veto_E = cms.double(0.07),
        DR_Veto_HO = cms.double(0.1)
    ),
    runArbitrationCleaner = cms.bool(True),
    fillEnergy = cms.bool(False),
    TrackerKinkFinderParameters = cms.PSet(
        DoPredictionsOnly = cms.bool(False),
        usePosition = cms.bool(True),
        diagonalOnly = cms.bool(False),
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        TrackerRecHitBuilder = cms.string('WithAngleAndTemplate'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        RefitDirection = cms.string('alongMomentum'),
        RefitRPCHits = cms.bool(True),
        Propagator = cms.string('SmartPropagatorAnyRKOpposite')
    ),
    TimingFillerParameters = cms.PSet(
        DTTimingParameters = cms.PSet(
            MatchParameters = cms.PSet(
                CSCsegments = cms.InputTag("cscSegments"),
                DTsegments = cms.InputTag("dt4DSegments"),
                DTradius = cms.double(0.01),
                TightMatchDT = cms.bool(False),
                TightMatchCSC = cms.bool(True)
            ),
            HitError = cms.double(6.0),
            DoWireCorr = cms.bool(True),
            PruneCut = cms.double(10000.0),
            DTsegments = cms.InputTag("dt4DSegments"),
            DropTheta = cms.bool(True),
            RequireBothProjections = cms.bool(False),
            HitsMin = cms.int32(3),
            DTTimeOffset = cms.double(0.0),
            debug = cms.bool(False),
            UseSegmentT0 = cms.bool(False),
            ServiceParameters = cms.PSet(
                Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny', 
                    'PropagatorWithMaterial', 
                    'PropagatorWithMaterialOpposite'),
                RPCLayers = cms.bool(True)
            )
        ),
        CSCTimingParameters = cms.PSet(
            MatchParameters = cms.PSet(
                CSCsegments = cms.InputTag("cscSegments"),
                DTsegments = cms.InputTag("dt4DSegments"),
                DTradius = cms.double(0.01),
                TightMatchDT = cms.bool(False),
                TightMatchCSC = cms.bool(True)
            ),
            CSCsegments = cms.InputTag("csc2DSegments"),
            CSCStripTimeOffset = cms.double(0.0),
            CSCStripError = cms.double(7.0),
            UseStripTime = cms.bool(True),
            debug = cms.bool(False),
            CSCWireError = cms.double(8.6),
            CSCWireTimeOffset = cms.double(0.0),
            ServiceParameters = cms.PSet(
                Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny', 
                    'PropagatorWithMaterial', 
                    'PropagatorWithMaterialOpposite'),
                RPCLayers = cms.bool(True)
            ),
            PruneCut = cms.double(9.0),
            UseWireTime = cms.bool(True)
        ),
        UseDT = cms.bool(True),
        EcalEnergyCut = cms.double(0.4),
        ErrorEB = cms.double(2.085),
        ErrorEE = cms.double(6.95),
        UseCSC = cms.bool(True),
        UseECAL = cms.bool(True)
    ),
    inputCollectionTypes = cms.vstring('inner tracks', 
        'outer tracks'),
    minCaloCompatibility = cms.double(0.6),
    ecalDepositName = cms.string('ecal'),
    minP = cms.double(3.0),
    fillIsolation = cms.bool(False),
    jetDepositName = cms.string('jets'),
    hoDepositName = cms.string('ho'),
    writeIsoDeposits = cms.bool(True),
    maxAbsPullX = cms.double(4.0),
    maxAbsPullY = cms.double(9999.0),
    minPt = cms.double(2.0),
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
        dRHcal = cms.double(9999.0),
        dRPreshowerPreselection = cms.double(0.2),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(False),
        dREcal = cms.double(9999.0),
        dREcalPreselection = cms.double(0.05),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
        propagateAllDirections = cms.bool(True),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(False),
        trajectoryUncertaintyTolerance = cms.double(-1.0),
        usePreshower = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(0.2),
        useMuon = cms.bool(True),
        useCalo = cms.bool(False),
        accountForTrajectoryChangeCalo = cms.bool(False),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(False)
    ),
    JetExtractorPSet = cms.PSet(
        PrintTimeReport = cms.untracked.bool(False),
        ExcludeMuonVeto = cms.bool(True),
        TrackAssociatorParameters = cms.PSet(
            muonMaxDistanceSigmaX = cms.double(0.0),
            muonMaxDistanceSigmaY = cms.double(0.0),
            CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
            dRHcal = cms.double(0.5),
            dRPreshowerPreselection = cms.double(0.2),
            CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
            useEcal = cms.bool(False),
            dREcal = cms.double(0.5),
            dREcalPreselection = cms.double(0.5),
            HORecHitCollectionLabel = cms.InputTag("horeco"),
            dRMuon = cms.double(9999.0),
            propagateAllDirections = cms.bool(True),
            muonMaxDistanceX = cms.double(5.0),
            muonMaxDistanceY = cms.double(5.0),
            useHO = cms.bool(False),
            trajectoryUncertaintyTolerance = cms.double(-1.0),
            usePreshower = cms.bool(False),
            DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
            EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
            dRHcalPreselection = cms.double(0.5),
            useMuon = cms.bool(False),
            useCalo = cms.bool(True),
            accountForTrajectoryChangeCalo = cms.bool(False),
            EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
            dRMuonPreselection = cms.double(0.2),
            truthMatch = cms.bool(False),
            HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
            useHcal = cms.bool(False)
        ),
        ServiceParameters = cms.PSet(
            Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny'),
            RPCLayers = cms.bool(False),
            UseMuonNavigation = cms.untracked.bool(False)
        ),
        ComponentName = cms.string('JetExtractor'),
        DR_Max = cms.double(1.0),
        PropagatorName = cms.string('SteppingHelixPropagatorAny'),
        JetCollectionLabel = cms.InputTag("ak4CaloJets"),
        DR_Veto = cms.double(0.1),
        Threshold = cms.double(5.0)
    ),
    fillGlobalTrackQuality = cms.bool(False),
    minPCaloMuon = cms.double(3.0),
    maxAbsDy = cms.double(9999.0),
    fillCaloCompatibility = cms.bool(False),
    fillMatching = cms.bool(True),
    MuonCaloCompatibility = cms.PSet(
        allSiPMHO = cms.bool(False),
        PionTemplateFileName = cms.FileInPath('RecoMuon/MuonIdentification/data/MuID_templates_pions_lowPt_3_1_norm.root'),
        MuonTemplateFileName = cms.FileInPath('RecoMuon/MuonIdentification/data/MuID_templates_muons_lowPt_3_1_norm.root'),
        delta_eta = cms.double(0.02),
        delta_phi = cms.double(0.02)
    ),
    fillTrackerKink = cms.bool(False),
    hcalDepositName = cms.string('hcal'),
    sigmaThresholdToFillCandidateP4WithGlobalFit = cms.double(2.0),
    inputCollectionLabels = cms.VInputTag(cms.InputTag("earlyGeneralTracks"), cms.InputTag("standAloneMuons","UpdatedAtVtx")),
    trackDepositName = cms.string('tracker'),
    maxAbsDx = cms.double(3.0),
    ptThresholdToFillCandidateP4WithGlobalFit = cms.double(200.0),
    minNumberOfMatches = cms.int32(1)
)


process.firstStepPrimaryVertices = cms.EDProducer("PrimaryVertexProducer",
    vertexCollections = cms.VPSet(cms.PSet(
        maxDistanceToBeam = cms.double(1.0),
        useBeamConstraint = cms.bool(False),
        minNdof = cms.double(0.0),
        algorithm = cms.string('AdaptiveVertexFitter'),
        label = cms.string('')
    ), 
        cms.PSet(
            maxDistanceToBeam = cms.double(1.0),
            useBeamConstraint = cms.bool(True),
            minNdof = cms.double(2.0),
            algorithm = cms.string('AdaptiveVertexFitter'),
            label = cms.string('WithBS')
        )),
    verbose = cms.untracked.bool(False),
    TkFilterParameters = cms.PSet(
        maxNormalizedChi2 = cms.double(20.0),
        minPt = cms.double(0.0),
        algorithm = cms.string('filter'),
        maxD0Significance = cms.double(5.0),
        trackQuality = cms.string('any'),
        minPixelLayersWithHits = cms.int32(2),
        minSiliconLayersWithHits = cms.int32(5)
    ),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    TrackLabel = cms.InputTag("initialStepTracks"),
    TkClusParameters = cms.PSet(
        TkDAClusParameters = cms.PSet(
            dzCutOff = cms.double(4.0),
            d0CutOff = cms.double(3.0),
            Tmin = cms.double(4.0),
            coolingFactor = cms.double(0.6),
            vertexSize = cms.double(0.01),
            use_vdt = cms.untracked.bool(True)
        ),
        algorithm = cms.string('DA_vect')
    )
)


process.generalTracks = cms.EDProducer("DuplicateListMerger",
    newQuality = cms.string('confirmed'),
    diffHitsCut = cms.int32(5),
    mergedMVAVals = cms.InputTag("duplicateTrackSelector","MVAVals"),
    originalSource = cms.InputTag("preDuplicateMergingGeneralTracks"),
    minTrkProbCut = cms.double(0.0),
    mergedSource = cms.InputTag("mergedDuplicateTracks"),
    candidateSource = cms.InputTag("duplicateTrackCandidates","candidateMap")
)


process.globalCombinedSeeds = cms.EDProducer("SeedCombiner",
    seedCollections = cms.VInputTag(cms.InputTag("globalSeedsFromTripletsWithVertices"), cms.InputTag("globalSeedsFromPairsWithVertices"))
)


process.globalMixedSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            originRadius = cms.double(0.2),
            ptMin = cms.double(0.9),
            originHalfLength = cms.double(21.2)
        ),
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        maxElement = cms.uint32(1000000),
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.InputTag("MixedLayerPairs")
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.globalPixelLessSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            originRadius = cms.double(0.2),
            ptMin = cms.double(0.9),
            originHalfLength = cms.double(40)
        ),
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(5000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        maxElement = cms.uint32(100000),
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.InputTag("pixelLessLayerPairs4PixelLessTracking")
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.globalPixelSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            originRadius = cms.double(0.2),
            ptMin = cms.double(0.9),
            originHalfLength = cms.double(21.2)
        ),
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.InputTag("PixelLayerPairs")
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.globalSeedsFromPairsWithVertices = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            useFakeVertices = cms.bool(False),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            useFixedError = cms.bool(True),
            originRadius = cms.double(0.2),
            sigmaZVertex = cms.double(3.0),
            fixedError = cms.double(0.2),
            VertexCollection = cms.InputTag("pixelVertices"),
            ptMin = cms.double(0.9),
            useFoundVertices = cms.bool(True),
            nSigmaZ = cms.double(4.0)
        ),
        ComponentName = cms.string('GlobalTrackingRegionWithVerticesProducer')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        maxElement = cms.uint32(1000000),
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.InputTag("MixedLayerPairs")
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.globalSeedsFromTriplets = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            originRadius = cms.double(0.2),
            ptMin = cms.double(0.9),
            originHalfLength = cms.double(21.2)
        ),
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        GeneratorPSet = cms.PSet(
            useBending = cms.bool(True),
            useFixedPreFiltering = cms.bool(False),
            maxElement = cms.uint32(1000000),
            SeedComparitorPSet = cms.PSet(
                ComponentName = cms.string('none')
            ),
            extraHitRPhitolerance = cms.double(0.032),
            useMultScattering = cms.bool(True),
            phiPreFiltering = cms.double(0.3),
            extraHitRZtolerance = cms.double(0.037),
            ComponentName = cms.string('PixelTripletHLTGenerator')
        ),
        SeedingLayers = cms.InputTag("PixelLayerTriplets")
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.initialStep = cms.EDProducer("TrackListMerger",
    ShareFrac = cms.double(0.19),
    writeOnlyTrkQuals = cms.bool(True),
    MinPT = cms.double(0.05),
    allowFirstHitShare = cms.bool(True),
    copyExtras = cms.untracked.bool(False),
    Epsilon = cms.double(-0.001),
    shareFrac = cms.double(0.99),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepSelector","initialStepV1"), cms.InputTag("initialStepSelector","initialStepV2"), cms.InputTag("initialStepSelector","initialStepV3")),
    indivShareFrac = cms.vdouble(1.0, 1.0, 1.0),
    MaxNormalizedChisq = cms.double(1000.0),
    hasSelector = cms.vint32(1, 1, 1),
    FoundHitBonus = cms.double(5.0),
    setsToMerge = cms.VPSet(cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0, 1, 2)
    )),
    MinFound = cms.int32(3),
    TrackProducers = cms.VInputTag(cms.InputTag("initialStepTracks"), cms.InputTag("initialStepTracks"), cms.InputTag("initialStepTracks")),
    LostHitPenalty = cms.double(20.0),
    newQuality = cms.string('confirmed')
)


process.initialStepSeedClusterMask = cms.EDProducer("SeedClusterRemover",
    trajectories = cms.InputTag("initialStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("pixelLessStepClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag(""),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.initialStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg'),
    MTOB = cms.PSet(

    ),
    TEC = cms.PSet(

    ),
    MTID = cms.PSet(

    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    MTEC = cms.PSet(

    ),
    MTIB = cms.PSet(

    ),
    TID = cms.PSet(

    ),
    TOB = cms.PSet(

    ),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    TIB = cms.PSet(

    )
)


process.initialStepSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        GeneratorPSet = cms.PSet(
            useBending = cms.bool(True),
            useFixedPreFiltering = cms.bool(False),
            maxElement = cms.uint32(1000000),
            SeedComparitorPSet = cms.PSet(
                clusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
                ComponentName = cms.string('LowPtClusterShapeSeedComparitor')
            ),
            extraHitRPhitolerance = cms.double(0.032),
            useMultScattering = cms.bool(True),
            phiPreFiltering = cms.double(0.3),
            extraHitRZtolerance = cms.double(0.037),
            ComponentName = cms.string('PixelTripletHLTGenerator')
        ),
        SeedingLayers = cms.InputTag("initialStepSeedLayers")
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originRadius = cms.double(0.02),
            nSigmaZ = cms.double(4.0),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            ptMin = cms.double(0.6)
        )
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.initialStepSelector = cms.EDProducer("MultiTrackSelector",
    src = cms.InputTag("initialStepTracks"),
    trackSelectors = cms.VPSet(cms.PSet(
        max_d0 = cms.double(100.0),
        minNumber3DLayers = cms.uint32(0),
        max_lostHitFraction = cms.double(1.0),
        applyAbsCutsIfNoPV = cms.bool(False),
        qualityBit = cms.string('loose'),
        minNumberLayers = cms.uint32(0),
        chi2n_par = cms.double(1.6),
        nSigmaZ = cms.double(4.0),
        dz_par2 = cms.vdouble(0.45, 4.0),
        applyAdaptedPVCuts = cms.bool(True),
        min_eta = cms.double(-9999.0),
        dz_par1 = cms.vdouble(0.65, 4.0),
        copyTrajectories = cms.untracked.bool(False),
        vtxNumber = cms.int32(-1),
        keepAllTracks = cms.bool(False),
        maxNumberLostLayers = cms.uint32(999),
        max_relpterr = cms.double(9999.0),
        copyExtras = cms.untracked.bool(True),
        vertexCut = cms.string('ndof>=2&!isFake'),
        max_z0 = cms.double(100.0),
        min_nhits = cms.uint32(0),
        name = cms.string('initialStepLoose'),
        max_minMissHitOutOrIn = cms.int32(99),
        chi2n_no1Dmod_par = cms.double(9999),
        res_par = cms.vdouble(0.003, 0.01),
        max_eta = cms.double(9999.0),
        d0_par2 = cms.vdouble(0.55, 4.0),
        d0_par1 = cms.vdouble(0.55, 4.0),
        preFilterName = cms.string(''),
        minHitsToBypassChecks = cms.uint32(20)
    ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(3),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(3),
            chi2n_par = cms.double(0.7),
            dz_par1 = cms.vdouble(0.35, 4.0),
            dz_par2 = cms.vdouble(0.4, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(2),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('initialStepTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('initialStepLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.4, 4.0),
            d0_par1 = cms.vdouble(0.3, 4.0),
            res_par = cms.vdouble(0.003, 0.01),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(3),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(3),
            chi2n_par = cms.double(0.7),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(0.4, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(0.35, 4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(2),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('initialStepV1'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.001),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.4, 4.0),
            d0_par1 = cms.vdouble(0.3, 4.0),
            preFilterName = cms.string('initialStepTight'),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(0),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(0),
            chi2n_par = cms.double(9999),
            dz_par1 = cms.vdouble(1.0, 3.0),
            dz_par2 = cms.vdouble(1.1, 3.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(False),
            maxNumberLostLayers = cms.uint32(999),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            minMVA = cms.double(0.5),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('initialStepV2'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string(''),
            useMVA = cms.bool(True),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.1, 3.0),
            d0_par1 = cms.vdouble(1.0, 3.0),
            res_par = cms.vdouble(0.003, 0.01),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(0),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(0),
            chi2n_par = cms.double(9999),
            dz_par1 = cms.vdouble(1.0, 4.0),
            dz_par2 = cms.vdouble(1.0, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(False),
            maxNumberLostLayers = cms.uint32(999),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            minMVA = cms.double(0.5),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('initialStepV3'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string(''),
            useMVA = cms.bool(True),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.0, 4.0),
            d0_par1 = cms.vdouble(1.0, 4.0),
            res_par = cms.vdouble(0.003, 0.01),
            minHitsToBypassChecks = cms.uint32(20)
        )),
    GBRForestLabel = cms.string('MVASelectorIter0_13TeV_v0'),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vertices = cms.InputTag("pixelVertices"),
    useVtxError = cms.bool(False),
    useAnyMVA = cms.bool(True),
    useVertices = cms.bool(True)
)


process.initialStepTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("initialStepSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('initialStepTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    numHitsForSeedCleaner = cms.int32(50)
)


process.initialStepTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("initialStepTrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('iter0'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.iter0TrackRefsForJets = cms.EDProducer("ChargedRefCandidateProducer",
    src = cms.InputTag("initialStepTracks"),
    particleType = cms.string('pi+')
)


process.jetCoreRegionalStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    FPix = cms.PSet(
        hitErrorRZ = cms.double(0.0036),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('siPixelRecHits'),
        useErrorsFromParam = cms.bool(True)
    ),
    layerList = cms.vstring('BPix1+BPix2', 
        'BPix1+BPix3', 
        'BPix2+BPix3', 
        'BPix1+FPix1_pos', 
        'BPix1+FPix1_neg', 
        'BPix2+FPix1_pos', 
        'BPix2+FPix1_neg', 
        'FPix1_pos+FPix2_pos', 
        'FPix1_neg+FPix2_neg', 
        'BPix3+TIB1', 
        'BPix3+TIB2'),
    BPix = cms.PSet(
        hitErrorRZ = cms.double(0.006),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('siPixelRecHits'),
        useErrorsFromParam = cms.bool(True)
    ),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)


process.jetCoreRegionalStepSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        maxElement = cms.uint32(1000000),
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.InputTag("jetCoreRegionalStepSeedLayers")
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('TauRegionalPixelSeedGenerator'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            deltaPhiRegion = cms.double(0.2),
            originHalfLength = cms.double(0.2),
            howToUseMeasurementTracker = cms.double(-1.0),
            measurementTrackerName = cms.string('MeasurementTrackerEvent'),
            deltaEtaRegion = cms.double(0.2),
            vertexSrc = cms.InputTag("firstStepGoodPrimaryVertices"),
            JetSrc = cms.InputTag("jetsForCoreTracking"),
            originRadius = cms.double(0.2),
            ptMin = cms.double(10.0)
        )
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial'),
        forceKinematicWithRegionDirection = cms.bool(True)
    )
)


process.jetCoreRegionalStepSelector = cms.EDProducer("MultiTrackSelector",
    src = cms.InputTag("jetCoreRegionalStepTracks"),
    trackSelectors = cms.VPSet(cms.PSet(
        max_d0 = cms.double(100.0),
        minNumber3DLayers = cms.uint32(0),
        max_lostHitFraction = cms.double(1.0),
        applyAbsCutsIfNoPV = cms.bool(False),
        qualityBit = cms.string('loose'),
        minNumberLayers = cms.uint32(0),
        chi2n_par = cms.double(1.6),
        nSigmaZ = cms.double(4.0),
        dz_par2 = cms.vdouble(0.45, 4.0),
        applyAdaptedPVCuts = cms.bool(True),
        min_eta = cms.double(-9999.0),
        dz_par1 = cms.vdouble(0.65, 4.0),
        copyTrajectories = cms.untracked.bool(False),
        vtxNumber = cms.int32(-1),
        keepAllTracks = cms.bool(False),
        maxNumberLostLayers = cms.uint32(999),
        max_relpterr = cms.double(9999.0),
        copyExtras = cms.untracked.bool(True),
        vertexCut = cms.string('ndof>=2&!isFake'),
        max_z0 = cms.double(100.0),
        min_nhits = cms.uint32(0),
        name = cms.string('jetCoreRegionalStepLoose'),
        max_minMissHitOutOrIn = cms.int32(99),
        chi2n_no1Dmod_par = cms.double(9999),
        res_par = cms.vdouble(0.003, 0.01),
        max_eta = cms.double(9999.0),
        d0_par2 = cms.vdouble(0.55, 4.0),
        d0_par1 = cms.vdouble(0.55, 4.0),
        preFilterName = cms.string(''),
        minHitsToBypassChecks = cms.uint32(20)
    ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(3),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(3),
            chi2n_par = cms.double(0.7),
            dz_par1 = cms.vdouble(0.35, 4.0),
            dz_par2 = cms.vdouble(0.4, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(2),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('jetCoreRegionalStepTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('jetCoreRegionalStepLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.4, 4.0),
            d0_par1 = cms.vdouble(0.3, 4.0),
            res_par = cms.vdouble(0.003, 0.01),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(3),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(3),
            chi2n_par = cms.double(0.7),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(0.4, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(0.35, 4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(2),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('jetCoreRegionalStep'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.001),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.4, 4.0),
            d0_par1 = cms.vdouble(0.3, 4.0),
            preFilterName = cms.string('jetCoreRegionalStepTight'),
            minHitsToBypassChecks = cms.uint32(20)
        )),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vertices = cms.InputTag("pixelVertices"),
    useVtxError = cms.bool(False),
    useVertices = cms.bool(True)
)


process.jetCoreRegionalStepTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("jetCoreRegionalStepSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(10000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('jetCoreRegionalStepTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder')
)


process.jetCoreRegionalStepTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("jetCoreRegionalStepTrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('iter7'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.lowPtTripletStepClusters = cms.EDProducer("TrackClusterRemover",
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
    trajectories = cms.InputTag("detachedTripletStepTracks"),
    oldClusterRemovalInfo = cms.InputTag("detachedTripletStepClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag("detachedTripletStep"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.lowPtTripletStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg'),
    MTOB = cms.PSet(

    ),
    TEC = cms.PSet(

    ),
    MTID = cms.PSet(

    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag("lowPtTripletStepClusters")
    ),
    MTEC = cms.PSet(

    ),
    MTIB = cms.PSet(

    ),
    TID = cms.PSet(

    ),
    TOB = cms.PSet(

    ),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag("lowPtTripletStepClusters")
    ),
    TIB = cms.PSet(

    )
)


process.lowPtTripletStepSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        GeneratorPSet = cms.PSet(
            useBending = cms.bool(True),
            useFixedPreFiltering = cms.bool(False),
            maxElement = cms.uint32(1000000),
            SeedComparitorPSet = cms.PSet(
                clusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
                ComponentName = cms.string('LowPtClusterShapeSeedComparitor')
            ),
            extraHitRPhitolerance = cms.double(0.032),
            useMultScattering = cms.bool(True),
            phiPreFiltering = cms.double(0.3),
            extraHitRZtolerance = cms.double(0.037),
            ComponentName = cms.string('PixelTripletHLTGenerator')
        ),
        SeedingLayers = cms.InputTag("lowPtTripletStepSeedLayers")
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originRadius = cms.double(0.02),
            nSigmaZ = cms.double(4.0),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            ptMin = cms.double(0.2)
        )
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.lowPtTripletStepSelector = cms.EDProducer("MultiTrackSelector",
    src = cms.InputTag("lowPtTripletStepTracks"),
    trackSelectors = cms.VPSet(cms.PSet(
        max_d0 = cms.double(100.0),
        minNumber3DLayers = cms.uint32(0),
        max_lostHitFraction = cms.double(1.0),
        applyAbsCutsIfNoPV = cms.bool(False),
        qualityBit = cms.string('loose'),
        minNumberLayers = cms.uint32(0),
        chi2n_par = cms.double(9999),
        nSigmaZ = cms.double(4.0),
        dz_par2 = cms.vdouble(0.45, 4.0),
        applyAdaptedPVCuts = cms.bool(True),
        min_eta = cms.double(-9999.0),
        dz_par1 = cms.vdouble(0.65, 4.0),
        copyTrajectories = cms.untracked.bool(False),
        vtxNumber = cms.int32(-1),
        keepAllTracks = cms.bool(False),
        maxNumberLostLayers = cms.uint32(999),
        max_relpterr = cms.double(9999.0),
        copyExtras = cms.untracked.bool(True),
        minMVA = cms.double(-0.6),
        vertexCut = cms.string('ndof>=2&!isFake'),
        max_z0 = cms.double(100.0),
        min_nhits = cms.uint32(0),
        name = cms.string('lowPtTripletStepLoose'),
        max_minMissHitOutOrIn = cms.int32(99),
        chi2n_no1Dmod_par = cms.double(9999),
        res_par = cms.vdouble(0.003, 0.01),
        useMVA = cms.bool(True),
        max_eta = cms.double(9999.0),
        d0_par2 = cms.vdouble(0.55, 4.0),
        d0_par1 = cms.vdouble(0.55, 4.0),
        preFilterName = cms.string(''),
        minHitsToBypassChecks = cms.uint32(20)
    ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(3),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(3),
            chi2n_par = cms.double(0.7),
            dz_par1 = cms.vdouble(0.35, 4.0),
            dz_par2 = cms.vdouble(0.4, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(2),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('lowPtTripletStepTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('lowPtTripletStepLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.4, 4.0),
            d0_par1 = cms.vdouble(0.3, 4.0),
            res_par = cms.vdouble(0.003, 0.01),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(0),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(0),
            chi2n_par = cms.double(9999),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(0.45, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(0.65, 4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(999),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            minMVA = cms.double(0.4),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('lowPtTripletStep'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.01),
            useMVA = cms.bool(True),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.55, 4.0),
            d0_par1 = cms.vdouble(0.55, 4.0),
            preFilterName = cms.string('lowPtTripletStepLoose'),
            minHitsToBypassChecks = cms.uint32(20)
        )),
    GBRForestLabel = cms.string('MVASelectorIter1_13TeV_v0'),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vertices = cms.InputTag("pixelVertices"),
    useVtxError = cms.bool(False),
    useAnyMVA = cms.bool(True),
    useVertices = cms.bool(True)
)


process.lowPtTripletStepTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("lowPtTripletStepSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('lowPtTripletStepTrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    numHitsForSeedCleaner = cms.int32(50),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('lowPtTripletStepTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    clustersToSkip = cms.InputTag("lowPtTripletStepClusters")
)


process.lowPtTripletStepTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("lowPtTripletStepTrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('iter1'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.mergedDuplicateTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("duplicateTrackCandidates","candidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('undefAlgorithm'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.mixedTripletStep = cms.EDProducer("TrackListMerger",
    ShareFrac = cms.double(0.19),
    writeOnlyTrkQuals = cms.bool(True),
    MinPT = cms.double(0.05),
    allowFirstHitShare = cms.bool(True),
    copyExtras = cms.untracked.bool(False),
    Epsilon = cms.double(-0.001),
    shareFrac = cms.double(0.11),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("mixedTripletStepSelector","mixedTripletStepVtx"), cms.InputTag("mixedTripletStepSelector","mixedTripletStepTrk")),
    indivShareFrac = cms.vdouble(0.11, 0.11),
    MaxNormalizedChisq = cms.double(1000.0),
    hasSelector = cms.vint32(1, 1),
    FoundHitBonus = cms.double(5.0),
    setsToMerge = cms.VPSet(cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0, 1)
    )),
    MinFound = cms.int32(3),
    TrackProducers = cms.VInputTag(cms.InputTag("mixedTripletStepTracks"), cms.InputTag("mixedTripletStepTracks")),
    LostHitPenalty = cms.double(20.0),
    newQuality = cms.string('confirmed')
)


process.mixedTripletStepClusters = cms.EDProducer("TrackClusterRemover",
    doStripChargeCheck = cms.bool(True),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
    trajectories = cms.InputTag("pixelPairStepTracks"),
    oldClusterRemovalInfo = cms.InputTag("pixelPairStepClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag("pixelPairStepSelector","pixelPairStep"),
    stripRecHits = cms.string('siStripMatchedRecHits'),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        minGoodStripCharge = cms.double(2069),
        maxChi2 = cms.double(9.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.mixedTripletStepSeedClusterMask = cms.EDProducer("SeedClusterRemover",
    trajectories = cms.InputTag("mixedTripletStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("pixelPairStepSeedClusterMask"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag(""),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.mixedTripletStepSeedLayersA = cms.EDProducer("SeedingLayersEDProducer",
    FPix = cms.PSet(
        skipClusters = cms.InputTag("mixedTripletStepClusters"),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg', 
        'BPix2+FPix1_pos+FPix2_pos', 
        'BPix2+FPix1_neg+FPix2_neg'),
    TEC = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        skipClusters = cms.InputTag("mixedTripletStepClusters"),
        maxRing = cms.int32(1)
    ),
    BPix = cms.PSet(
        skipClusters = cms.InputTag("mixedTripletStepClusters"),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('siPixelRecHits')
    )
)


process.mixedTripletStepSeedLayersB = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix2+BPix3+TIB1'),
    BPix = cms.PSet(
        skipClusters = cms.InputTag("mixedTripletStepClusters"),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    TIB = cms.PSet(
        skipClusters = cms.InputTag("mixedTripletStepClusters"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)


process.mixedTripletStepSeeds = cms.EDProducer("SeedCombiner",
    seedCollections = cms.VInputTag(cms.InputTag("mixedTripletStepSeedsA"), cms.InputTag("mixedTripletStepSeedsB"))
)


process.mixedTripletStepSeedsA = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        GeneratorPSet = cms.PSet(
            useBending = cms.bool(True),
            useFixedPreFiltering = cms.bool(False),
            maxElement = cms.uint32(100000),
            ComponentName = cms.string('PixelTripletLargeTipGenerator'),
            extraHitRPhitolerance = cms.double(0.0),
            useMultScattering = cms.bool(True),
            phiPreFiltering = cms.double(0.3),
            extraHitRZtolerance = cms.double(0.0)
        ),
        SeedingLayers = cms.InputTag("mixedTripletStepSeedLayersA")
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterStripHits = cms.bool(True),
        ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
        ClusterShapeHitFilterName = cms.string('mixedTripletStepClusterShapeHitFilter'),
        FilterPixelHits = cms.bool(True),
        FilterAtHelixStage = cms.bool(False)
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            originRadius = cms.double(1.5),
            ptMin = cms.double(0.4),
            originHalfLength = cms.double(15.0)
        ),
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsTripletOnlyCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.mixedTripletStepSeedsB = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        GeneratorPSet = cms.PSet(
            useBending = cms.bool(True),
            useFixedPreFiltering = cms.bool(False),
            maxElement = cms.uint32(100000),
            ComponentName = cms.string('PixelTripletLargeTipGenerator'),
            extraHitRPhitolerance = cms.double(0.0),
            useMultScattering = cms.bool(True),
            phiPreFiltering = cms.double(0.3),
            extraHitRZtolerance = cms.double(0.0)
        ),
        SeedingLayers = cms.InputTag("mixedTripletStepSeedLayersB")
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterStripHits = cms.bool(True),
        ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
        ClusterShapeHitFilterName = cms.string('mixedTripletStepClusterShapeHitFilter'),
        FilterPixelHits = cms.bool(True),
        FilterAtHelixStage = cms.bool(False)
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            originRadius = cms.double(1.5),
            ptMin = cms.double(0.6),
            originHalfLength = cms.double(10.0)
        ),
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsTripletOnlyCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.mixedTripletStepSelector = cms.EDProducer("MultiTrackSelector",
    src = cms.InputTag("mixedTripletStepTracks"),
    trackSelectors = cms.VPSet(cms.PSet(
        max_d0 = cms.double(100.0),
        minNumber3DLayers = cms.uint32(0),
        max_lostHitFraction = cms.double(1.0),
        applyAbsCutsIfNoPV = cms.bool(False),
        qualityBit = cms.string('loose'),
        minNumberLayers = cms.uint32(0),
        chi2n_par = cms.double(9999),
        nSigmaZ = cms.double(4.0),
        dz_par2 = cms.vdouble(1.3, 3.0),
        applyAdaptedPVCuts = cms.bool(True),
        min_eta = cms.double(-9999.0),
        dz_par1 = cms.vdouble(1.2, 3.0),
        copyTrajectories = cms.untracked.bool(False),
        vtxNumber = cms.int32(-1),
        keepAllTracks = cms.bool(False),
        maxNumberLostLayers = cms.uint32(999),
        max_relpterr = cms.double(9999.0),
        copyExtras = cms.untracked.bool(True),
        minMVA = cms.double(-0.5),
        vertexCut = cms.string('ndof>=2&!isFake'),
        max_z0 = cms.double(100.0),
        min_nhits = cms.uint32(0),
        name = cms.string('mixedTripletStepVtxLoose'),
        max_minMissHitOutOrIn = cms.int32(99),
        chi2n_no1Dmod_par = cms.double(9999),
        res_par = cms.vdouble(0.003, 0.01),
        useMVA = cms.bool(True),
        max_eta = cms.double(9999.0),
        d0_par2 = cms.vdouble(1.3, 3.0),
        d0_par1 = cms.vdouble(1.2, 3.0),
        preFilterName = cms.string(''),
        minHitsToBypassChecks = cms.uint32(20)
    ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(0),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('loose'),
            minNumberLayers = cms.uint32(0),
            chi2n_par = cms.double(9999),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(1.1, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(1.1, 4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(False),
            maxNumberLostLayers = cms.uint32(999),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            minMVA = cms.double(-0.5),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('mixedTripletStepTrkLoose'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.01),
            useMVA = cms.bool(True),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.1, 4.0),
            d0_par1 = cms.vdouble(1.1, 4.0),
            preFilterName = cms.string(''),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(3),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(3),
            chi2n_par = cms.double(0.6),
            dz_par1 = cms.vdouble(1.1, 3.0),
            dz_par2 = cms.vdouble(1.2, 3.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(1),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('mixedTripletStepVtxTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('mixedTripletStepVtxLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.2, 3.0),
            d0_par1 = cms.vdouble(1.1, 3.0),
            res_par = cms.vdouble(0.003, 0.001),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(4),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(5),
            chi2n_par = cms.double(0.4),
            dz_par1 = cms.vdouble(1.0, 4.0),
            dz_par2 = cms.vdouble(1.0, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(1),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('mixedTripletStepTrkTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('mixedTripletStepTrkLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.0, 4.0),
            d0_par1 = cms.vdouble(1.0, 4.0),
            res_par = cms.vdouble(0.003, 0.001),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(0),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(0),
            chi2n_par = cms.double(9999),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(1.1, 3.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(1.0, 3.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(999),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            minMVA = cms.double(0.5),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('mixedTripletStepVtx'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.01),
            useMVA = cms.bool(True),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.1, 3.0),
            d0_par1 = cms.vdouble(1.0, 3.0),
            preFilterName = cms.string('mixedTripletStepVtxLoose'),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(0),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(0),
            chi2n_par = cms.double(9999),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(0.8, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(0.8, 4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(999),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            minMVA = cms.double(0.5),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('mixedTripletStepTrk'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.01),
            useMVA = cms.bool(True),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.8, 4.0),
            d0_par1 = cms.vdouble(0.8, 4.0),
            preFilterName = cms.string('mixedTripletStepTrkLoose'),
            minHitsToBypassChecks = cms.uint32(20)
        )),
    GBRForestLabel = cms.string('MVASelectorIter4_13TeV_v0'),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vertices = cms.InputTag("pixelVertices"),
    useVtxError = cms.bool(False),
    useAnyMVA = cms.bool(True),
    useVertices = cms.bool(True)
)


process.mixedTripletStepTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("mixedTripletStepSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('mixedTripletStepTrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('mixedTripletStepTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    numHitsForSeedCleaner = cms.int32(50),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    clustersToSkip = cms.InputTag("mixedTripletStepClusters")
)


process.mixedTripletStepTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("mixedTripletStepTrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('iter4'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.muonSeededSeedsInOut = cms.EDProducer("MuonReSeeder",
    DoPredictionsOnly = cms.bool(False),
    insideOut = cms.bool(True),
    cut = cms.string('pt > 2'),
    src = cms.InputTag("earlyMuons"),
    Fitter = cms.string('KFFitterForRefitInsideOut'),
    TrackerRecHitBuilder = cms.string('WithAngleAndTemplate'),
    Smoother = cms.string('KFSmootherForRefitInsideOut'),
    MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
    RefitDirection = cms.string('alongMomentum'),
    RefitRPCHits = cms.bool(True),
    debug = cms.untracked.bool(False),
    Propagator = cms.string('SmartPropagatorAnyRKOpposite'),
    layersToKeep = cms.int32(5)
)


process.muonSeededSeedsInOutAsTracks = cms.EDProducer("FakeTrackProducerFromSeed",
    src = cms.InputTag("muonSeededSeedsInOut")
)


process.muonSeededSeedsOutIn = cms.EDProducer("OutsideInMuonSeeder",
    fromVertex = cms.bool(True),
    hitsToTry = cms.int32(3),
    layersToTry = cms.int32(3),
    hitCollector = cms.string('hitCollectorForOutInMuonSeeds'),
    errorRescaleFactor = cms.double(2.0),
    src = cms.InputTag("earlyMuons"),
    muonPropagator = cms.string('SteppingHelixPropagatorAlong'),
    cut = cms.string('pt > 10 && outerTrack.hitPattern.muonStationsWithValidHits >= 2'),
    maxEtaForTOB = cms.double(1.8),
    minEtaForTEC = cms.double(0.7),
    debug = cms.untracked.bool(False),
    trackerPropagator = cms.string('PropagatorWithMaterial')
)


process.muonSeededSeedsOutInAsTracks = cms.EDProducer("FakeTrackProducerFromSeed",
    src = cms.InputTag("muonSeededSeedsOutIn")
)


process.muonSeededTrackCandidatesInOut = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("muonSeededSeedsInOut"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('muonSeededTrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('none'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('muonSeededTrajectoryBuilderForInOut')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder')
)


process.muonSeededTrackCandidatesInOutAsTracks = cms.EDProducer("FakeTrackProducerFromCandidate",
    src = cms.InputTag("muonSeededTrackCandidatesInOut")
)


process.muonSeededTrackCandidatesOutIn = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("muonSeededSeedsOutIn"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('muonSeededTrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('muonSeededTrajectoryBuilderForOutIn')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    onlyPixelHitsForSeedCleaner = cms.bool(False),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    numHitsForSeedCleaner = cms.int32(50)
)


process.muonSeededTrackCandidatesOutInAsTracks = cms.EDProducer("FakeTrackProducerFromCandidate",
    src = cms.InputTag("muonSeededTrackCandidatesOutIn")
)


process.muonSeededTracksInOut = cms.EDProducer("TrackProducer",
    src = cms.InputTag("muonSeededTrackCandidatesInOut"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('muonSeededFittingSmootherWithOutliersRejectionAndRK'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('iter9'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.muonSeededTracksInOutSelector = cms.EDProducer("MultiTrackSelector",
    src = cms.InputTag("muonSeededTracksInOut"),
    trackSelectors = cms.VPSet(cms.PSet(
        max_d0 = cms.double(100.0),
        minNumber3DLayers = cms.uint32(0),
        max_lostHitFraction = cms.double(1.0),
        applyAbsCutsIfNoPV = cms.bool(False),
        qualityBit = cms.string('loose'),
        minNumberLayers = cms.uint32(3),
        chi2n_par = cms.double(10.0),
        nSigmaZ = cms.double(4.0),
        dz_par2 = cms.vdouble(0.45, 4.0),
        applyAdaptedPVCuts = cms.bool(False),
        min_eta = cms.double(-9999.0),
        dz_par1 = cms.vdouble(0.65, 4.0),
        copyTrajectories = cms.untracked.bool(False),
        vtxNumber = cms.int32(-1),
        keepAllTracks = cms.bool(False),
        maxNumberLostLayers = cms.uint32(4),
        max_relpterr = cms.double(9999.0),
        copyExtras = cms.untracked.bool(True),
        vertexCut = cms.string('ndof>=2&!isFake'),
        max_z0 = cms.double(100.0),
        min_nhits = cms.uint32(5),
        name = cms.string('muonSeededTracksInOutLoose'),
        max_minMissHitOutOrIn = cms.int32(99),
        chi2n_no1Dmod_par = cms.double(9999),
        res_par = cms.vdouble(0.003, 0.01),
        max_eta = cms.double(9999.0),
        d0_par2 = cms.vdouble(0.55, 4.0),
        d0_par1 = cms.vdouble(0.55, 4.0),
        preFilterName = cms.string(''),
        minHitsToBypassChecks = cms.uint32(7)
    ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(2),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(5),
            chi2n_par = cms.double(1.0),
            dz_par1 = cms.vdouble(0.35, 4.0),
            dz_par2 = cms.vdouble(0.4, 4.0),
            applyAdaptedPVCuts = cms.bool(False),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(3),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(6),
            name = cms.string('muonSeededTracksInOutTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('muonSeededTracksInOutLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.4, 4.0),
            d0_par1 = cms.vdouble(0.3, 4.0),
            res_par = cms.vdouble(0.003, 0.01),
            minHitsToBypassChecks = cms.uint32(10)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(2),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(5),
            chi2n_par = cms.double(0.4),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(0.4, 4.0),
            applyAdaptedPVCuts = cms.bool(False),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(0.35, 4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(2),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(7),
            name = cms.string('muonSeededTracksInOutHighPurity'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.001),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.4, 4.0),
            d0_par1 = cms.vdouble(0.3, 4.0),
            preFilterName = cms.string('muonSeededTracksInOutTight'),
            minHitsToBypassChecks = cms.uint32(20)
        )),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vertices = cms.InputTag("pixelVertices"),
    useVtxError = cms.bool(False),
    useVertices = cms.bool(True)
)


process.muonSeededTracksOutIn = cms.EDProducer("TrackProducer",
    src = cms.InputTag("muonSeededTrackCandidatesOutIn"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('muonSeededFittingSmootherWithOutliersRejectionAndRK'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('iter10'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.muonSeededTracksOutInSelector = cms.EDProducer("MultiTrackSelector",
    src = cms.InputTag("muonSeededTracksOutIn"),
    trackSelectors = cms.VPSet(cms.PSet(
        max_d0 = cms.double(100.0),
        minNumber3DLayers = cms.uint32(0),
        max_lostHitFraction = cms.double(1.0),
        applyAbsCutsIfNoPV = cms.bool(False),
        qualityBit = cms.string('loose'),
        minNumberLayers = cms.uint32(3),
        chi2n_par = cms.double(10.0),
        nSigmaZ = cms.double(4.0),
        dz_par2 = cms.vdouble(0.45, 4.0),
        applyAdaptedPVCuts = cms.bool(False),
        min_eta = cms.double(-9999.0),
        dz_par1 = cms.vdouble(0.65, 4.0),
        copyTrajectories = cms.untracked.bool(False),
        vtxNumber = cms.int32(-1),
        keepAllTracks = cms.bool(False),
        maxNumberLostLayers = cms.uint32(4),
        max_relpterr = cms.double(9999.0),
        copyExtras = cms.untracked.bool(True),
        vertexCut = cms.string('ndof>=2&!isFake'),
        max_z0 = cms.double(100.0),
        min_nhits = cms.uint32(5),
        name = cms.string('muonSeededTracksOutInLoose'),
        max_minMissHitOutOrIn = cms.int32(99),
        chi2n_no1Dmod_par = cms.double(9999),
        res_par = cms.vdouble(0.003, 0.01),
        max_eta = cms.double(9999.0),
        d0_par2 = cms.vdouble(0.55, 4.0),
        d0_par1 = cms.vdouble(0.55, 4.0),
        preFilterName = cms.string(''),
        minHitsToBypassChecks = cms.uint32(7)
    ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(2),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(5),
            chi2n_par = cms.double(1.0),
            dz_par1 = cms.vdouble(0.35, 4.0),
            dz_par2 = cms.vdouble(0.4, 4.0),
            applyAdaptedPVCuts = cms.bool(False),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(3),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(6),
            name = cms.string('muonSeededTracksOutInTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('muonSeededTracksOutInLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.4, 4.0),
            d0_par1 = cms.vdouble(0.3, 4.0),
            res_par = cms.vdouble(0.003, 0.01),
            minHitsToBypassChecks = cms.uint32(10)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(2),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(5),
            chi2n_par = cms.double(0.4),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(0.4, 4.0),
            applyAdaptedPVCuts = cms.bool(False),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(0.35, 4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(2),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(7),
            name = cms.string('muonSeededTracksOutInHighPurity'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.001),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.4, 4.0),
            d0_par1 = cms.vdouble(0.3, 4.0),
            preFilterName = cms.string('muonSeededTracksOutInTight'),
            minHitsToBypassChecks = cms.uint32(20)
        )),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vertices = cms.InputTag("pixelVertices"),
    useVtxError = cms.bool(False),
    useVertices = cms.bool(True)
)


process.newCombinedSeeds = cms.EDProducer("SeedCombiner",
    seedCollections = cms.VInputTag(cms.InputTag("initialStepSeeds"), cms.InputTag("pixelPairStepSeeds"), cms.InputTag("mixedTripletStepSeeds"), cms.InputTag("pixelLessStepSeeds"), cms.InputTag("tripletElectronSeeds"), 
        cms.InputTag("pixelPairElectronSeeds"), cms.InputTag("stripPairElectronSeeds"))
)


process.offlineBeamSpot = cms.EDProducer("BeamSpotProducer")


process.photonConvTrajSeedFromQuadruplets = cms.EDProducer("PhotonConversionTrajectorySeedProducerFromQuadruplets",
    beamSpotInputTag = cms.InputTag("offlineBeamSpot"),
    QuadCutPSet = cms.PSet(
        Cut_maxLegPt = cms.double(10.0),
        Cut_zCA = cms.double(100),
        apply_zCACut = cms.bool(False),
        Cut_BeamPipeRadius = cms.double(3.0),
        apply_DeltaPhiCuts = cms.bool(True),
        rejectAllQuads = cms.bool(False),
        Cut_minLegPt = cms.double(0.6),
        apply_Arbitration = cms.bool(True),
        Cut_DeltaRho = cms.double(12.0),
        apply_ClusterShapeFilter = cms.bool(True)
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterStripHits = cms.bool(True),
        ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        FilterPixelHits = cms.bool(False),
        FilterAtHelixStage = cms.bool(True)
    ),
    ClusterCheckPSet = cms.PSet(
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(50000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters"),
        MaxNumberOfPixelClusters = cms.uint32(10000)
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originRadius = cms.double(3.0),
            ptMin = cms.double(0.2),
            originHalfLength = cms.double(12.0),
            beamSpot = cms.InputTag("offlineBeamSpot")
        )
    ),
    DoxcheckSeedCandidates = cms.bool(False),
    xcheckSeedCandidates = cms.string('xcheckSeedCandidates'),
    SeedCreatorPSet = cms.PSet(
        ComponentName = cms.string('SeedForPhotonConversionFromQuadruplets'),
        SeedMomentumForBOFF = cms.double(5.0),
        propagator = cms.string('PropagatorWithMaterial')
    ),
    TrackRefitter = cms.InputTag("generalTracks"),
    OrderedHitsFactoryPSet = cms.PSet(
        maxElement = cms.uint32(900000),
        SeedingLayers = cms.InputTag("conv2LayerPairs")
    ),
    primaryVerticesTag = cms.InputTag("pixelVertices"),
    newSeedCandidates = cms.string('conv2SeedCandidates')
)


process.photonConvTrajSeedFromSingleLeg = cms.EDProducer("PhotonConversionTrajectorySeedProducerFromSingleLeg",
    vtxMinDoF = cms.double(4),
    beamSpotInputTag = cms.InputTag("offlineBeamSpot"),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originRadius = cms.double(3.0),
            ptMin = cms.double(0.2),
            originHalfLength = cms.double(12.0),
            beamSpot = cms.InputTag("offlineBeamSpot")
        )
    ),
    DoxcheckSeedCandidates = cms.bool(False),
    xcheckSeedCandidates = cms.string('xcheckSeedCandidates'),
    SeedCreatorPSet = cms.PSet(
        ComponentName = cms.string('SeedForPhotonConversion1Leg'),
        SeedMomentumForBOFF = cms.double(5.0),
        propagator = cms.string('PropagatorWithMaterial')
    ),
    TrackRefitter = cms.InputTag("generalTracks"),
    OrderedHitsFactoryPSet = cms.PSet(
        maxElement = cms.uint32(40000),
        SeedingLayers = cms.InputTag("convLayerPairs"),
        maxHitPairsPerTrackAndGenerator = cms.uint32(10)
    ),
    applyTkVtxConstraint = cms.bool(True),
    maxDZSigmas = cms.double(10.0),
    maxNumSelVtx = cms.uint32(2),
    primaryVerticesTag = cms.InputTag("pixelVertices"),
    newSeedCandidates = cms.string('convSeedCandidates')
)


process.pixelLessLayerPairs4PixelLessTracking = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('TIB1+TIB2', 
        'TIB1+TIB3', 
        'TIB2+TIB3', 
        'TIB1+TID1_pos', 
        'TIB1+TID1_neg', 
        'TIB2+TID1_pos', 
        'TIB2+TID1_neg', 
        'TIB1+TID2_pos', 
        'TIB1+TID2_neg', 
        'TID1_pos+TID2_pos', 
        'TID2_pos+TID3_pos', 
        'TID3_pos+TEC2_pos', 
        'TEC1_pos+TEC2_pos', 
        'TEC2_pos+TEC3_pos', 
        'TID1_neg+TID2_neg', 
        'TID2_neg+TID3_neg', 
        'TID3_neg+TEC2_neg', 
        'TEC1_neg+TEC2_neg', 
        'TEC2_neg+TEC3_neg'),
    MTOB = cms.PSet(

    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
        maxRing = cms.int32(2),
        stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched")
    ),
    MTID = cms.PSet(

    ),
    FPix = cms.PSet(

    ),
    MTEC = cms.PSet(

    ),
    MTIB = cms.PSet(

    ),
    TID = cms.PSet(

    ),
    TOB = cms.PSet(

    ),
    BPix = cms.PSet(

    ),
    TIB = cms.PSet(

    ),
    TID1 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
        maxRing = cms.int32(3),
        minRing = cms.int32(1)
    ),
    TID2 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
        maxRing = cms.int32(3),
        minRing = cms.int32(1)
    ),
    TID3 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
        maxRing = cms.int32(2),
        minRing = cms.int32(1)
    ),
    TIB1 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useSimpleRphiHitsCleaner = cms.bool(False),
        stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched")
    ),
    TIB2 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useSimpleRphiHitsCleaner = cms.bool(False),
        stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched")
    ),
    TIB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        useSimpleRphiHitsCleaner = cms.bool(False),
        stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched")
    )
)


process.pixelLessStep = cms.EDProducer("TrackListMerger",
    ShareFrac = cms.double(0.19),
    writeOnlyTrkQuals = cms.bool(True),
    MinPT = cms.double(0.05),
    allowFirstHitShare = cms.bool(True),
    copyExtras = cms.untracked.bool(False),
    Epsilon = cms.double(-0.001),
    shareFrac = cms.double(0.11),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("pixelLessStepSelector","pixelLessStep"), cms.InputTag("pixelLessStepSelector","pixelLessStepVtx"), cms.InputTag("pixelLessStepSelector","pixelLessStepTrk")),
    indivShareFrac = cms.vdouble(0.11, 0.11, 0.11),
    MaxNormalizedChisq = cms.double(1000.0),
    hasSelector = cms.vint32(1, 1, 1),
    FoundHitBonus = cms.double(5.0),
    setsToMerge = cms.VPSet(cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0, 1, 2)
    )),
    MinFound = cms.int32(3),
    TrackProducers = cms.VInputTag(cms.InputTag("pixelLessStepTracks"), cms.InputTag("pixelLessStepTracks"), cms.InputTag("pixelLessStepTracks")),
    LostHitPenalty = cms.double(20.0),
    newQuality = cms.string('confirmed')
)


process.pixelLessStepClusters = cms.EDProducer("TrackClusterRemover",
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
    trajectories = cms.InputTag("mixedTripletStepTracks"),
    oldClusterRemovalInfo = cms.InputTag("mixedTripletStepClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag("mixedTripletStep"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.pixelLessStepSeedClusterMask = cms.EDProducer("SeedClusterRemover",
    trajectories = cms.InputTag("pixelLessStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("mixedTripletStepSeedClusterMask"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag(""),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.pixelLessStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('TIB1+TIB2+MTIB3', 
        'TIB1+TIB2+MTID1_pos', 
        'TIB1+TIB2+MTID1_neg', 
        'TID1_pos+TID2_pos+TID3_pos', 
        'TID1_neg+TID2_neg+TID3_neg', 
        'TID1_pos+TID2_pos+MTID3_pos', 
        'TID1_neg+TID2_neg+MTID3_neg', 
        'TID1_pos+TID2_pos+MTEC1_pos', 
        'TID1_neg+TID2_neg+MTEC1_neg', 
        'TID2_pos+TID3_pos+TEC1_pos', 
        'TID2_neg+TID3_neg+TEC1_neg', 
        'TID2_pos+TID3_pos+MTEC1_pos', 
        'TID2_neg+TID3_neg+MTEC1_neg', 
        'TEC1_pos+TEC2_pos+TEC3_pos', 
        'TEC1_neg+TEC2_neg+TEC3_neg', 
        'TEC1_pos+TEC2_pos+MTEC3_pos', 
        'TEC1_neg+TEC2_neg+MTEC3_neg', 
        'TEC1_pos+TEC2_pos+TEC4_pos', 
        'TEC1_neg+TEC2_neg+TEC4_neg', 
        'TEC1_pos+TEC2_pos+MTEC4_pos', 
        'TEC1_neg+TEC2_neg+MTEC4_neg', 
        'TEC2_pos+TEC3_pos+TEC4_pos', 
        'TEC2_neg+TEC3_neg+TEC4_neg', 
        'TEC2_pos+TEC3_pos+MTEC4_pos', 
        'TEC2_neg+TEC3_neg+MTEC4_neg', 
        'TEC2_pos+TEC3_pos+TEC5_pos', 
        'TEC2_neg+TEC3_neg+TEC5_neg', 
        'TEC2_pos+TEC3_pos+TEC6_pos', 
        'TEC2_neg+TEC3_neg+TEC6_neg', 
        'TEC3_pos+TEC4_pos+TEC5_pos', 
        'TEC3_neg+TEC4_neg+TEC5_neg', 
        'TEC3_pos+TEC4_pos+MTEC5_pos', 
        'TEC3_neg+TEC4_neg+MTEC5_neg', 
        'TEC3_pos+TEC5_pos+TEC6_pos', 
        'TEC3_neg+TEC5_neg+TEC6_neg', 
        'TEC4_pos+TEC5_pos+TEC6_pos', 
        'TEC4_neg+TEC5_neg+TEC6_neg'),
    TEC = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        skipClusters = cms.InputTag("pixelLessStepClusters"),
        maxRing = cms.int32(2)
    ),
    MTID = cms.PSet(
        minRing = cms.int32(3),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        skipClusters = cms.InputTag("pixelLessStepClusters"),
        maxRing = cms.int32(3)
    ),
    MTEC = cms.PSet(
        minRing = cms.int32(3),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        skipClusters = cms.InputTag("pixelLessStepClusters"),
        maxRing = cms.int32(3)
    ),
    MTIB = cms.PSet(
        skipClusters = cms.InputTag("pixelLessStepClusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TID = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        skipClusters = cms.InputTag("pixelLessStepClusters"),
        maxRing = cms.int32(2)
    ),
    TIB = cms.PSet(
        skipClusters = cms.InputTag("pixelLessStepClusters"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)


process.pixelLessStepSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardMultiHitGenerator'),
        GeneratorPSet = cms.PSet(
            detIdsToDebug = cms.vint32(0, 0, 0),
            pt_interv = cms.vdouble(0.4, 0.7, 1.0, 2.0),
            useFixedPreFiltering = cms.bool(False),
            refitHits = cms.bool(True),
            chi2VsPtCut = cms.bool(True),
            maxChi2 = cms.double(5.0),
            extraHitRPhitolerance = cms.double(0.0),
            extraRKDBox = cms.double(0.2),
            chi2_cuts = cms.vdouble(3.0, 4.0, 5.0, 5.0),
            extraZKDBox = cms.double(0.2),
            extraPhiKDBox = cms.double(0.005),
            maxElement = cms.uint32(100000),
            TTRHBuilder = cms.string('WithTrackAngle'),
            phiPreFiltering = cms.double(0.3),
            debug = cms.bool(False),
            extraHitRZtolerance = cms.double(0.0),
            ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
            ComponentName = cms.string('MultiHitGeneratorFromChi2'),
            fnSigmaRZ = cms.double(2.0)
        ),
        SeedingLayers = cms.InputTag("pixelLessStepSeedLayers")
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterStripHits = cms.bool(True),
        ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
        ClusterShapeHitFilterName = cms.string('pixelLessStepClusterShapeHitFilter'),
        FilterPixelHits = cms.bool(False),
        FilterAtHelixStage = cms.bool(True)
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            originRadius = cms.double(1.0),
            ptMin = cms.double(0.4),
            originHalfLength = cms.double(12.0)
        ),
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsTripletOnlyCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.pixelLessStepSelector = cms.EDProducer("MultiTrackSelector",
    src = cms.InputTag("pixelLessStepTracks"),
    trackSelectors = cms.VPSet(cms.PSet(
        max_d0 = cms.double(100.0),
        minNumber3DLayers = cms.uint32(0),
        max_lostHitFraction = cms.double(1.0),
        applyAbsCutsIfNoPV = cms.bool(False),
        qualityBit = cms.string('loose'),
        minNumberLayers = cms.uint32(0),
        chi2n_par = cms.double(9999),
        nSigmaZ = cms.double(4.0),
        dz_par2 = cms.vdouble(1.2, 4.0),
        applyAdaptedPVCuts = cms.bool(True),
        min_eta = cms.double(-9999.0),
        dz_par1 = cms.vdouble(1.2, 4.0),
        copyTrajectories = cms.untracked.bool(False),
        vtxNumber = cms.int32(-1),
        keepAllTracks = cms.bool(False),
        maxNumberLostLayers = cms.uint32(999),
        max_relpterr = cms.double(9999.0),
        copyExtras = cms.untracked.bool(True),
        minMVA = cms.double(0.4),
        vertexCut = cms.string('ndof>=2&!isFake'),
        max_z0 = cms.double(100.0),
        min_nhits = cms.uint32(0),
        name = cms.string('pixelLessStepLoose'),
        max_minMissHitOutOrIn = cms.int32(99),
        chi2n_no1Dmod_par = cms.double(9999),
        res_par = cms.vdouble(0.003, 0.01),
        useMVA = cms.bool(True),
        max_eta = cms.double(9999.0),
        d0_par2 = cms.vdouble(1.2, 4.0),
        d0_par1 = cms.vdouble(1.2, 4.0),
        preFilterName = cms.string(''),
        minHitsToBypassChecks = cms.uint32(20)
    ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(3),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(4),
            chi2n_par = cms.double(0.3),
            dz_par1 = cms.vdouble(0.9, 4.0),
            dz_par2 = cms.vdouble(0.9, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(0),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('pixelLessStepTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('pixelLessStepLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.9, 4.0),
            d0_par1 = cms.vdouble(0.9, 4.0),
            res_par = cms.vdouble(0.003, 0.001),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(0),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(0),
            chi2n_par = cms.double(9999),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(0.7, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(0.7, 4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(999),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            minMVA = cms.double(0.4),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('pixelLessStep'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.01),
            useMVA = cms.bool(True),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.7, 4.0),
            d0_par1 = cms.vdouble(0.7, 4.0),
            preFilterName = cms.string('pixelLessStepLoose'),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(0),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(0),
            chi2n_par = cms.double(9999),
            dz_par1 = cms.vdouble(1.0, 3.0),
            dz_par2 = cms.vdouble(1.1, 3.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(False),
            maxNumberLostLayers = cms.uint32(999),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            minMVA = cms.double(0.5),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('pixelLessStepVtx'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string(''),
            useMVA = cms.bool(True),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.1, 3.0),
            d0_par1 = cms.vdouble(1.0, 3.0),
            res_par = cms.vdouble(0.003, 0.01),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(0),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(0),
            chi2n_par = cms.double(9999),
            dz_par1 = cms.vdouble(0.8, 4.0),
            dz_par2 = cms.vdouble(0.8, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(False),
            maxNumberLostLayers = cms.uint32(999),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            minMVA = cms.double(0.5),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('pixelLessStepTrk'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string(''),
            useMVA = cms.bool(True),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.8, 4.0),
            d0_par1 = cms.vdouble(0.8, 4.0),
            res_par = cms.vdouble(0.003, 0.01),
            minHitsToBypassChecks = cms.uint32(20)
        )),
    GBRForestLabel = cms.string('MVASelectorIter5_13TeV_v0'),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vertices = cms.InputTag("pixelVertices"),
    useVtxError = cms.bool(False),
    useAnyMVA = cms.bool(True),
    useVertices = cms.bool(True)
)


process.pixelLessStepTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("pixelLessStepSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('pixelLessStepTrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('pixelLessStepTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    numHitsForSeedCleaner = cms.int32(50),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    clustersToSkip = cms.InputTag("pixelLessStepClusters")
)


process.pixelLessStepTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("pixelLessStepTrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('iter5'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.pixelPairElectronSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    FPix = cms.PSet(
        skipClusters = cms.InputTag("tripletElectronClusterMask"),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    layerList = cms.vstring('BPix1+BPix2', 
        'BPix1+BPix3', 
        'BPix2+BPix3', 
        'BPix1+FPix1_pos', 
        'BPix1+FPix1_neg', 
        'BPix1+FPix2_pos', 
        'BPix1+FPix2_neg', 
        'BPix2+FPix1_pos', 
        'BPix2+FPix1_neg', 
        'FPix1_pos+FPix2_pos', 
        'FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        skipClusters = cms.InputTag("tripletElectronClusterMask"),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits')
    )
)


process.pixelPairElectronSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        maxElement = cms.uint32(1000000),
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.InputTag("pixelPairElectronSeedLayers")
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            useFakeVertices = cms.bool(False),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            useFixedError = cms.bool(True),
            originRadius = cms.double(0.015),
            sigmaZVertex = cms.double(3.0),
            fixedError = cms.double(0.03),
            VertexCollection = cms.InputTag("pixelVertices"),
            ptMin = cms.double(1.0),
            useFoundVertices = cms.bool(True),
            nSigmaZ = cms.double(4.0)
        ),
        ComponentName = cms.string('GlobalTrackingRegionWithVerticesProducer')
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.pixelPairStepClusters = cms.EDProducer("TrackClusterRemover",
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
    trajectories = cms.InputTag("lowPtTripletStepTracks"),
    oldClusterRemovalInfo = cms.InputTag("lowPtTripletStepClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.pixelPairStepSeedClusterMask = cms.EDProducer("SeedClusterRemover",
    trajectories = cms.InputTag("pixelPairStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("initialStepSeedClusterMask"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag(""),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.pixelPairStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    FPix = cms.PSet(
        skipClusters = cms.InputTag("pixelPairStepClusters"),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    layerList = cms.vstring('BPix1+BPix2', 
        'BPix1+BPix3', 
        'BPix2+BPix3', 
        'BPix1+FPix1_pos', 
        'BPix1+FPix1_neg', 
        'BPix2+FPix1_pos', 
        'BPix2+FPix1_neg', 
        'FPix1_pos+FPix2_pos', 
        'FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        skipClusters = cms.InputTag("pixelPairStepClusters"),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits')
    )
)


process.pixelPairStepSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        maxElement = cms.uint32(1000000),
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.InputTag("pixelPairStepSeedLayers")
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterStripHits = cms.bool(False),
        ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        FilterPixelHits = cms.bool(True),
        FilterAtHelixStage = cms.bool(True)
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            useFakeVertices = cms.bool(False),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            useFixedError = cms.bool(True),
            originRadius = cms.double(0.015),
            sigmaZVertex = cms.double(3.0),
            fixedError = cms.double(0.03),
            VertexCollection = cms.InputTag("pixelVertices"),
            ptMin = cms.double(0.6),
            useFoundVertices = cms.bool(True),
            nSigmaZ = cms.double(4.0)
        ),
        ComponentName = cms.string('GlobalTrackingRegionWithVerticesProducer')
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.pixelPairStepSelector = cms.EDProducer("MultiTrackSelector",
    src = cms.InputTag("pixelPairStepTracks"),
    trackSelectors = cms.VPSet(cms.PSet(
        max_d0 = cms.double(100.0),
        minNumber3DLayers = cms.uint32(0),
        max_lostHitFraction = cms.double(1.0),
        applyAbsCutsIfNoPV = cms.bool(False),
        qualityBit = cms.string('loose'),
        minNumberLayers = cms.uint32(0),
        chi2n_par = cms.double(1.6),
        nSigmaZ = cms.double(4.0),
        dz_par2 = cms.vdouble(0.45, 4.0),
        applyAdaptedPVCuts = cms.bool(True),
        min_eta = cms.double(-9999.0),
        dz_par1 = cms.vdouble(0.65, 4.0),
        copyTrajectories = cms.untracked.bool(False),
        vtxNumber = cms.int32(-1),
        keepAllTracks = cms.bool(False),
        maxNumberLostLayers = cms.uint32(999),
        max_relpterr = cms.double(9999.0),
        copyExtras = cms.untracked.bool(True),
        vertexCut = cms.string('ndof>=2&!isFake'),
        max_z0 = cms.double(100.0),
        min_nhits = cms.uint32(0),
        name = cms.string('pixelPairStepLoose'),
        max_minMissHitOutOrIn = cms.int32(99),
        chi2n_no1Dmod_par = cms.double(9999),
        res_par = cms.vdouble(0.003, 0.01),
        max_eta = cms.double(9999.0),
        d0_par2 = cms.vdouble(0.55, 4.0),
        d0_par1 = cms.vdouble(0.55, 4.0),
        preFilterName = cms.string(''),
        minHitsToBypassChecks = cms.uint32(20)
    ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(3),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(3),
            chi2n_par = cms.double(0.7),
            dz_par1 = cms.vdouble(0.35, 4.0),
            dz_par2 = cms.vdouble(0.4, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(2),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('pixelPairStepTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('pixelPairStepLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.4, 4.0),
            d0_par1 = cms.vdouble(0.3, 4.0),
            res_par = cms.vdouble(0.003, 0.01),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(3),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(3),
            chi2n_par = cms.double(0.7),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(0.4, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(0.35, 4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(2),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('pixelPairStep'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.001),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(0.4, 4.0),
            d0_par1 = cms.vdouble(0.3, 4.0),
            preFilterName = cms.string('pixelPairStepTight'),
            minHitsToBypassChecks = cms.uint32(20)
        )),
    GBRForestLabel = cms.string('MVASelectorIter2_13TeV_v0'),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vertices = cms.InputTag("pixelVertices"),
    useVtxError = cms.bool(False),
    useAnyMVA = cms.bool(True),
    useVertices = cms.bool(True)
)


process.pixelPairStepTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("pixelPairStepSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('pixelPairStepTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    clustersToSkip = cms.InputTag("pixelPairStepClusters"),
    numHitsForSeedCleaner = cms.int32(50)
)


process.pixelPairStepTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("pixelPairStepTrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('iter2'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.preDuplicateMergingGeneralTracks = cms.EDProducer("TrackListMerger",
    ShareFrac = cms.double(0.19),
    writeOnlyTrkQuals = cms.bool(False),
    MinPT = cms.double(0.05),
    indivShareFrac = cms.vdouble(1.0, 0.19, 0.16, 0.19, 0.13, 
        0.11, 0.11, 0.09),
    copyExtras = cms.untracked.bool(True),
    Epsilon = cms.double(-0.001),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("muonSeededTracksInOutSelector","muonSeededTracksInOutHighPurity"), cms.InputTag("muonSeededTracksInOutSelector","muonSeededTracksInOutHighPurity"), cms.InputTag("muonSeededTracksOutInSelector","muonSeededTracksOutInHighPurity")),
    allowFirstHitShare = cms.bool(True),
    mvaValueTags = cms.VInputTag(cms.InputTag("earlyGeneralTracks","MVAVals"), cms.InputTag("muonSeededTracksInOutSelector","MVAVals"), cms.InputTag("muonSeededTracksOutInSelector","MVAVals")),
    makeReKeyedSeeds = cms.untracked.bool(False),
    MaxNormalizedChisq = cms.double(1000.0),
    FoundHitBonus = cms.double(100.0),
    setsToMerge = cms.VPSet(cms.PSet(
        pQual = cms.bool(False),
        tLists = cms.vint32(0, 1, 2)
    )),
    MinFound = cms.int32(3),
    hasSelector = cms.vint32(0, 1, 1),
    TrackProducers = cms.VInputTag(cms.InputTag("earlyGeneralTracks"), cms.InputTag("muonSeededTracksInOut"), cms.InputTag("muonSeededTracksOutIn")),
    LostHitPenalty = cms.double(1.0),
    newQuality = cms.string('confirmed')
)


process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")


process.regionalCosmicCkfTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("regionalCosmicTrackerSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('GroupedCkfTrajectoryBuilderP5')
    ),
    NavigationSchool = cms.string('CosmicNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder')
)


process.regionalCosmicTrackerSeedingLayers = cms.EDProducer("SeedingLayersEDProducer",
    TOB = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(6),
        maxRing = cms.int32(7)
    ),
    layerList = cms.vstring('TOB6+TOB5', 
        'TOB6+TOB4', 
        'TOB6+TOB3', 
        'TOB5+TOB4', 
        'TOB5+TOB3', 
        'TOB4+TOB3', 
        'TEC1_neg+TOB6', 
        'TEC1_neg+TOB5', 
        'TEC1_neg+TOB4', 
        'TEC1_pos+TOB6', 
        'TEC1_pos+TOB5', 
        'TEC1_pos+TOB4')
)


process.regionalCosmicTrackerSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(10000),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(10000),
        ClusterCollectionLabel = cms.InputTag("siStripClusters"),
        doClusterCheck = cms.bool(False)
    ),
    RegionFactoryPSet = cms.PSet(
        CollectionsPSet = cms.PSet(
            recoMuonsCollection = cms.InputTag(""),
            recoTrackMuonsCollection = cms.InputTag("cosmicMuons"),
            recoL2MuonsCollection = cms.InputTag("")
        ),
        ComponentName = cms.string('CosmicRegionalSeedGenerator'),
        RegionInJetsCheckPSet = cms.PSet(
            recoCaloJetsCollection = cms.InputTag("ak4CaloJets"),
            deltaRExclusionSize = cms.double(0.3),
            jetsPtMin = cms.double(5),
            doJetsExclusionCheck = cms.bool(True)
        ),
        ToolsPSet = cms.PSet(
            regionBase = cms.string('seedOnCosmicMuon'),
            thePropagatorName = cms.string('AnalyticalPropagator')
        ),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            deltaPhiRegion = cms.double(0.1),
            measurementTrackerName = cms.string(''),
            zVertex = cms.double(5),
            deltaEtaRegion = cms.double(0.1),
            ptMin = cms.double(1.0),
            rVertex = cms.double(5)
        )
    ),
    SeedCreatorPSet = cms.PSet(
        ComponentName = cms.string('CosmicSeedCreator'),
        maxseeds = cms.int32(10000),
        propagator = cms.string('PropagatorWithMaterial')
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('GenericPairGenerator'),
        LayerSrc = cms.InputTag("regionalCosmicTrackerSeedingLayers")
    ),
    TTRHBuilder = cms.string('WithTrackAngle'),
    RegionInJetsCheckPSet = cms.PSet(
        doJetsExclusionCheck = cms.bool(False)
    )
)


process.regionalCosmicTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("regionalCosmicCkfTrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('FittingSmootherRKP5'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    GeometricInnerState = cms.bool(True),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    AlgorithmName = cms.string('ctf'),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.seedClusterRemover = cms.EDProducer("SeedClusterRemover",
    trajectories = cms.InputTag("initialStepSeeds"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag(""),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.seedingLayersEDProducer = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring(),
    MTOB = cms.PSet(

    ),
    TEC = cms.PSet(

    ),
    MTID = cms.PSet(

    ),
    FPix = cms.PSet(

    ),
    MTEC = cms.PSet(

    ),
    MTIB = cms.PSet(

    ),
    TID = cms.PSet(

    ),
    TOB = cms.PSet(

    ),
    BPix = cms.PSet(

    ),
    TIB = cms.PSet(

    )
)


process.siPixelClusters = cms.EDProducer("SiPixelClusterProducer",
    src = cms.InputTag("siPixelDigis"),
    ChannelThreshold = cms.int32(1000),
    maxNumberOfClusters = cms.int32(-1),
    SplitClusters = cms.bool(False),
    MissCalibrate = cms.untracked.bool(True),
    VCaltoElectronGain = cms.int32(65),
    VCaltoElectronOffset = cms.int32(-414),
    payloadType = cms.string('Offline'),
    SeedThreshold = cms.int32(1000),
    ClusterThreshold = cms.double(4000.0)
)


process.siPixelClustersBottom = cms.EDProducer("PixelClusterSelectorTopBottom",
    y = cms.double(-1),
    label = cms.InputTag("siPixelClusters")
)


process.siPixelClustersTop = cms.EDProducer("PixelClusterSelectorTopBottom",
    y = cms.double(1),
    label = cms.InputTag("siPixelClusters")
)


process.siPixelRecHits = cms.EDProducer("SiPixelRecHitConverter",
    VerboseLevel = cms.untracked.int32(0),
    src = cms.InputTag("siPixelClusters"),
    CPE = cms.string('PixelCPEGeneric')
)


process.siPixelRecHitsBottom = cms.EDProducer("SiPixelRecHitConverter",
    VerboseLevel = cms.untracked.int32(0),
    src = cms.InputTag("siPixelClustersBottom"),
    CPE = cms.string('PixelCPEGeneric')
)


process.siPixelRecHitsTop = cms.EDProducer("SiPixelRecHitConverter",
    VerboseLevel = cms.untracked.int32(0),
    src = cms.InputTag("siPixelClustersTop"),
    CPE = cms.string('PixelCPEGeneric')
)


process.siStripClusters = cms.EDProducer("SiStripClusterizer",
    DigiProducersList = cms.VInputTag(cms.InputTag("siStripDigis","ZeroSuppressed"), cms.InputTag("siStripZeroSuppression","VirginRaw"), cms.InputTag("siStripZeroSuppression","ProcessedRaw"), cms.InputTag("siStripZeroSuppression","ScopeMode")),
    Clusterizer = cms.PSet(
        ChannelThreshold = cms.double(2.0),
        MaxSequentialBad = cms.uint32(1),
        Algorithm = cms.string('ThreeThresholdAlgorithm'),
        MaxSequentialHoles = cms.uint32(0),
        MaxAdjacentBad = cms.uint32(0),
        QualityLabel = cms.string(''),
        SeedThreshold = cms.double(3.0),
        RemoveApvShots = cms.bool(True),
        ClusterThreshold = cms.double(5.0)
    )
)


process.siStripClustersBottom = cms.EDProducer("StripClusterSelectorTopBottom",
    y = cms.double(-1),
    label = cms.InputTag("siStripClusters")
)


process.siStripClustersTop = cms.EDProducer("StripClusterSelectorTopBottom",
    y = cms.double(1),
    label = cms.InputTag("siStripClusters")
)


process.siStripMatchedRecHits = cms.EDProducer("SiStripRecHitConverter",
    StripCPE = cms.ESInputTag("StripCPEfromTrackAngleESProducer","StripCPEfromTrackAngle"),
    stereoRecHits = cms.string('stereoRecHit'),
    useSiStripQuality = cms.bool(False),
    matchedRecHits = cms.string('matchedRecHit'),
    ClusterProducer = cms.InputTag("siStripClusters"),
    VerbosityLevel = cms.untracked.int32(1),
    rphiRecHits = cms.string('rphiRecHit'),
    Matcher = cms.ESInputTag("SiStripRecHitMatcherESProducer","StandardMatcher"),
    siStripQualityLabel = cms.ESInputTag(""),
    MaskBadAPVFibers = cms.bool(False)
)


process.siStripMatchedRecHitsBottom = cms.EDProducer("SiStripRecHitConverter",
    StripCPE = cms.ESInputTag("StripCPEfromTrackAngleESProducer","StripCPEfromTrackAngle"),
    stereoRecHits = cms.string('stereoRecHit'),
    useSiStripQuality = cms.bool(False),
    matchedRecHits = cms.string('matchedRecHit'),
    ClusterProducer = cms.InputTag("siStripClustersBottom"),
    VerbosityLevel = cms.untracked.int32(1),
    rphiRecHits = cms.string('rphiRecHit'),
    Matcher = cms.ESInputTag("SiStripRecHitMatcherESProducer","StandardMatcher"),
    siStripQualityLabel = cms.ESInputTag(""),
    MaskBadAPVFibers = cms.bool(False)
)


process.siStripMatchedRecHitsTop = cms.EDProducer("SiStripRecHitConverter",
    StripCPE = cms.ESInputTag("StripCPEfromTrackAngleESProducer","StripCPEfromTrackAngle"),
    stereoRecHits = cms.string('stereoRecHit'),
    useSiStripQuality = cms.bool(False),
    matchedRecHits = cms.string('matchedRecHit'),
    ClusterProducer = cms.InputTag("siStripClustersTop"),
    VerbosityLevel = cms.untracked.int32(1),
    rphiRecHits = cms.string('rphiRecHit'),
    Matcher = cms.ESInputTag("SiStripRecHitMatcherESProducer","StandardMatcher"),
    siStripQualityLabel = cms.ESInputTag(""),
    MaskBadAPVFibers = cms.bool(False)
)


process.siStripZeroSuppression = cms.EDProducer("SiStripZeroSuppression",
    fixCM = cms.bool(False),
    DigisToMergeVR = cms.InputTag("siStripVRDigis","VirginRaw"),
    produceCalculatedBaseline = cms.bool(False),
    produceRawDigis = cms.bool(True),
    RawDigiProducersList = cms.VInputTag(cms.InputTag("siStripDigis","VirginRaw"), cms.InputTag("siStripDigis","ProcessedRaw"), cms.InputTag("siStripDigis","ScopeMode")),
    storeInZScollBadAPV = cms.bool(True),
    mergeCollections = cms.bool(False),
    Algorithms = cms.PSet(
        CutToAvoidSignal = cms.double(2.0),
        slopeY = cms.int32(4),
        slopeX = cms.int32(3),
        PedestalSubtractionFedMode = cms.bool(False),
        Fraction = cms.double(0.2),
        minStripsToFit = cms.uint32(4),
        consecThreshold = cms.uint32(5),
        hitStripThreshold = cms.uint32(40),
        Deviation = cms.uint32(25),
        CommonModeNoiseSubtractionMode = cms.string('IteratedMedian'),
        filteredBaselineDerivativeSumSquare = cms.double(30),
        ApplyBaselineCleaner = cms.bool(True),
        doAPVRestore = cms.bool(True),
        TruncateInSuppressor = cms.bool(True),
        restoreThreshold = cms.double(0.5),
        APVInspectMode = cms.string('BaselineFollower'),
        ForceNoRestore = cms.bool(False),
        useRealMeanCM = cms.bool(False),
        ApplyBaselineRejection = cms.bool(True),
        DeltaCMThreshold = cms.uint32(20),
        nSigmaNoiseDerTh = cms.uint32(4),
        nSaturatedStrip = cms.uint32(2),
        SiStripFedZeroSuppressionMode = cms.uint32(4),
        useCMMeanMap = cms.bool(False),
        SelfSelectRestoreAlgo = cms.bool(False),
        distortionThreshold = cms.uint32(20),
        filteredBaselineMax = cms.double(6),
        Iterations = cms.int32(3),
        CleaningSequence = cms.uint32(1),
        nSmooth = cms.uint32(9),
        APVRestoreMode = cms.string('BaselineFollower'),
        MeanCM = cms.int32(0)
    ),
    DigisToMergeZS = cms.InputTag("siStripDigis","ZeroSuppressed"),
    storeCM = cms.bool(True),
    produceBaselinePoints = cms.bool(False)
)


process.simpleCosmicBONSeedingLayers = cms.EDProducer("SeedingLayersEDProducer",
    TOB5 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB4 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB6 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(7)
    ),
    TIB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TIB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    layerList = cms.vstring('TOB4+TOB5+TOB6', 
        'TOB3+TOB5+TOB6', 
        'TOB3+TOB4+TOB5', 
        'TOB3+TOB4+TOB6', 
        'TOB2+TOB4+TOB5', 
        'TOB2+TOB3+TOB5', 
        'TEC7_pos+TEC8_pos+TEC9_pos', 
        'TEC6_pos+TEC7_pos+TEC8_pos', 
        'TEC5_pos+TEC6_pos+TEC7_pos', 
        'TEC4_pos+TEC5_pos+TEC6_pos', 
        'TEC3_pos+TEC4_pos+TEC5_pos', 
        'TEC2_pos+TEC3_pos+TEC4_pos', 
        'TEC1_pos+TEC2_pos+TEC3_pos', 
        'TEC7_neg+TEC8_neg+TEC9_neg', 
        'TEC6_neg+TEC7_neg+TEC8_neg', 
        'TEC5_neg+TEC6_neg+TEC7_neg', 
        'TEC4_neg+TEC5_neg+TEC6_neg', 
        'TEC3_neg+TEC4_neg+TEC5_neg', 
        'TEC2_neg+TEC3_neg+TEC4_neg', 
        'TEC1_neg+TEC2_neg+TEC3_neg', 
        'TEC6_pos+TEC8_pos+TEC9_pos', 
        'TEC5_pos+TEC7_pos+TEC8_pos', 
        'TEC4_pos+TEC6_pos+TEC7_pos', 
        'TEC3_pos+TEC5_pos+TEC6_pos', 
        'TEC2_pos+TEC4_pos+TEC5_pos', 
        'TEC1_pos+TEC3_pos+TEC4_pos', 
        'TEC6_pos+TEC7_pos+TEC9_pos', 
        'TEC5_pos+TEC6_pos+TEC8_pos', 
        'TEC4_pos+TEC5_pos+TEC7_pos', 
        'TEC3_pos+TEC4_pos+TEC6_pos', 
        'TEC2_pos+TEC3_pos+TEC5_pos', 
        'TEC1_pos+TEC2_pos+TEC4_pos', 
        'TEC6_neg+TEC8_neg+TEC9_neg', 
        'TEC5_neg+TEC7_neg+TEC8_neg', 
        'TEC4_neg+TEC6_neg+TEC7_neg', 
        'TEC3_neg+TEC5_neg+TEC6_neg', 
        'TEC2_neg+TEC4_neg+TEC5_neg', 
        'TEC1_neg+TEC3_neg+TEC4_neg', 
        'TEC6_neg+TEC7_neg+TEC9_neg', 
        'TEC5_neg+TEC6_neg+TEC8_neg', 
        'TEC4_neg+TEC5_neg+TEC7_neg', 
        'TEC3_neg+TEC4_neg+TEC6_neg', 
        'TEC2_neg+TEC3_neg+TEC5_neg', 
        'TEC1_neg+TEC2_neg+TEC4_neg', 
        'TOB6+TEC1_pos+TEC2_pos', 
        'TOB6+TEC1_neg+TEC2_neg', 
        'TOB6+TOB5+TEC1_pos', 
        'TOB6+TOB5+TEC1_neg')
)


process.simpleCosmicBONSeedingLayersBottom = cms.EDProducer("SeedingLayersEDProducer",
    TOB5 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    TOB4 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB6 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    TOB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    TOB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit"),
        maxRing = cms.int32(7)
    ),
    TIB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TIB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
    ),
    layerList = cms.vstring('TOB4+TOB5+TOB6', 
        'TOB3+TOB5+TOB6', 
        'TOB3+TOB4+TOB5', 
        'TOB3+TOB4+TOB6', 
        'TOB2+TOB4+TOB5', 
        'TOB2+TOB3+TOB5', 
        'TEC7_pos+TEC8_pos+TEC9_pos', 
        'TEC6_pos+TEC7_pos+TEC8_pos', 
        'TEC5_pos+TEC6_pos+TEC7_pos', 
        'TEC4_pos+TEC5_pos+TEC6_pos', 
        'TEC3_pos+TEC4_pos+TEC5_pos', 
        'TEC2_pos+TEC3_pos+TEC4_pos', 
        'TEC1_pos+TEC2_pos+TEC3_pos', 
        'TEC7_neg+TEC8_neg+TEC9_neg', 
        'TEC6_neg+TEC7_neg+TEC8_neg', 
        'TEC5_neg+TEC6_neg+TEC7_neg', 
        'TEC4_neg+TEC5_neg+TEC6_neg', 
        'TEC3_neg+TEC4_neg+TEC5_neg', 
        'TEC2_neg+TEC3_neg+TEC4_neg', 
        'TEC1_neg+TEC2_neg+TEC3_neg', 
        'TEC6_pos+TEC8_pos+TEC9_pos', 
        'TEC5_pos+TEC7_pos+TEC8_pos', 
        'TEC4_pos+TEC6_pos+TEC7_pos', 
        'TEC3_pos+TEC5_pos+TEC6_pos', 
        'TEC2_pos+TEC4_pos+TEC5_pos', 
        'TEC1_pos+TEC3_pos+TEC4_pos', 
        'TEC6_pos+TEC7_pos+TEC9_pos', 
        'TEC5_pos+TEC6_pos+TEC8_pos', 
        'TEC4_pos+TEC5_pos+TEC7_pos', 
        'TEC3_pos+TEC4_pos+TEC6_pos', 
        'TEC2_pos+TEC3_pos+TEC5_pos', 
        'TEC1_pos+TEC2_pos+TEC4_pos', 
        'TEC6_neg+TEC8_neg+TEC9_neg', 
        'TEC5_neg+TEC7_neg+TEC8_neg', 
        'TEC4_neg+TEC6_neg+TEC7_neg', 
        'TEC3_neg+TEC5_neg+TEC6_neg', 
        'TEC2_neg+TEC4_neg+TEC5_neg', 
        'TEC1_neg+TEC3_neg+TEC4_neg', 
        'TEC6_neg+TEC7_neg+TEC9_neg', 
        'TEC5_neg+TEC6_neg+TEC8_neg', 
        'TEC4_neg+TEC5_neg+TEC7_neg', 
        'TEC3_neg+TEC4_neg+TEC6_neg', 
        'TEC2_neg+TEC3_neg+TEC5_neg', 
        'TEC1_neg+TEC2_neg+TEC4_neg', 
        'TOB6+TEC1_pos+TEC2_pos', 
        'TOB6+TEC1_neg+TEC2_neg', 
        'TOB6+TOB5+TEC1_pos', 
        'TOB6+TOB5+TEC1_neg')
)


process.simpleCosmicBONSeedingLayersTop = cms.EDProducer("SeedingLayersEDProducer",
    TOB5 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    TOB4 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB6 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    TOB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    TOB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(False),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit"),
        maxRing = cms.int32(7)
    ),
    TIB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TIB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
    ),
    layerList = cms.vstring('TOB4+TOB5+TOB6', 
        'TOB3+TOB5+TOB6', 
        'TOB3+TOB4+TOB5', 
        'TOB3+TOB4+TOB6', 
        'TOB2+TOB4+TOB5', 
        'TOB2+TOB3+TOB5', 
        'TEC7_pos+TEC8_pos+TEC9_pos', 
        'TEC6_pos+TEC7_pos+TEC8_pos', 
        'TEC5_pos+TEC6_pos+TEC7_pos', 
        'TEC4_pos+TEC5_pos+TEC6_pos', 
        'TEC3_pos+TEC4_pos+TEC5_pos', 
        'TEC2_pos+TEC3_pos+TEC4_pos', 
        'TEC1_pos+TEC2_pos+TEC3_pos', 
        'TEC7_neg+TEC8_neg+TEC9_neg', 
        'TEC6_neg+TEC7_neg+TEC8_neg', 
        'TEC5_neg+TEC6_neg+TEC7_neg', 
        'TEC4_neg+TEC5_neg+TEC6_neg', 
        'TEC3_neg+TEC4_neg+TEC5_neg', 
        'TEC2_neg+TEC3_neg+TEC4_neg', 
        'TEC1_neg+TEC2_neg+TEC3_neg', 
        'TEC6_pos+TEC8_pos+TEC9_pos', 
        'TEC5_pos+TEC7_pos+TEC8_pos', 
        'TEC4_pos+TEC6_pos+TEC7_pos', 
        'TEC3_pos+TEC5_pos+TEC6_pos', 
        'TEC2_pos+TEC4_pos+TEC5_pos', 
        'TEC1_pos+TEC3_pos+TEC4_pos', 
        'TEC6_pos+TEC7_pos+TEC9_pos', 
        'TEC5_pos+TEC6_pos+TEC8_pos', 
        'TEC4_pos+TEC5_pos+TEC7_pos', 
        'TEC3_pos+TEC4_pos+TEC6_pos', 
        'TEC2_pos+TEC3_pos+TEC5_pos', 
        'TEC1_pos+TEC2_pos+TEC4_pos', 
        'TEC6_neg+TEC8_neg+TEC9_neg', 
        'TEC5_neg+TEC7_neg+TEC8_neg', 
        'TEC4_neg+TEC6_neg+TEC7_neg', 
        'TEC3_neg+TEC5_neg+TEC6_neg', 
        'TEC2_neg+TEC4_neg+TEC5_neg', 
        'TEC1_neg+TEC3_neg+TEC4_neg', 
        'TEC6_neg+TEC7_neg+TEC9_neg', 
        'TEC5_neg+TEC6_neg+TEC8_neg', 
        'TEC4_neg+TEC5_neg+TEC7_neg', 
        'TEC3_neg+TEC4_neg+TEC6_neg', 
        'TEC2_neg+TEC3_neg+TEC5_neg', 
        'TEC1_neg+TEC2_neg+TEC4_neg', 
        'TOB6+TEC1_pos+TEC2_pos', 
        'TOB6+TEC1_neg+TEC2_neg', 
        'TOB6+TOB5+TEC1_pos', 
        'TOB6+TOB5+TEC1_neg')
)


process.simpleCosmicBONSeeds = cms.EDProducer("SimpleCosmicBONSeeder",
    TripletsSrc = cms.InputTag("simpleCosmicBONSeedingLayers"),
    helixDebugLevel = cms.untracked.uint32(0),
    seedOnMiddle = cms.bool(False),
    RegionPSet = cms.PSet(
        pMin = cms.double(1.0),
        originRadius = cms.double(150.0),
        ptMin = cms.double(0.5),
        originZPosition = cms.double(0.0),
        originHalfLength = cms.double(90.0)
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(300),
        DontCountDetsAboveNClusters = cms.uint32(20),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(300),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    HitsPerModuleCheck = cms.PSet(
        checkHitsPerModule = cms.bool(True),
        Thresholds = cms.PSet(
            TOB = cms.int32(20),
            TID = cms.int32(20),
            TEC = cms.int32(20),
            TIB = cms.int32(20)
        )
    ),
    minimumGoodHitsInSeed = cms.int32(3),
    seedDebugLevel = cms.untracked.uint32(0),
    TripletsDebugLevel = cms.untracked.uint32(0),
    NegativeYOnly = cms.bool(False),
    maxTriplets = cms.int32(50000),
    ClusterChargeCheck = cms.PSet(
        Thresholds = cms.PSet(
            TOB = cms.int32(0),
            TID = cms.int32(0),
            TEC = cms.int32(0),
            TIB = cms.int32(0)
        ),
        matchedRecHitsUseAnd = cms.bool(True),
        checkCharge = cms.bool(False)
    ),
    PositiveYOnly = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    writeTriplets = cms.bool(False),
    maxSeeds = cms.int32(20000),
    rescaleError = cms.double(1.0)
)


process.simpleCosmicBONSeedsBottom = cms.EDProducer("SimpleCosmicBONSeeder",
    TripletsSrc = cms.InputTag("simpleCosmicBONSeedingLayersBottom"),
    helixDebugLevel = cms.untracked.uint32(0),
    seedOnMiddle = cms.bool(False),
    RegionPSet = cms.PSet(
        pMin = cms.double(1.0),
        originRadius = cms.double(150.0),
        ptMin = cms.double(0.5),
        originZPosition = cms.double(0.0),
        originHalfLength = cms.double(90.0)
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(300),
        DontCountDetsAboveNClusters = cms.uint32(20),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(150),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClustersBottom")
    ),
    HitsPerModuleCheck = cms.PSet(
        checkHitsPerModule = cms.bool(True),
        Thresholds = cms.PSet(
            TOB = cms.int32(20),
            TID = cms.int32(20),
            TEC = cms.int32(20),
            TIB = cms.int32(20)
        )
    ),
    minimumGoodHitsInSeed = cms.int32(3),
    seedDebugLevel = cms.untracked.uint32(0),
    TripletsDebugLevel = cms.untracked.uint32(0),
    NegativeYOnly = cms.bool(True),
    maxTriplets = cms.int32(50000),
    ClusterChargeCheck = cms.PSet(
        Thresholds = cms.PSet(
            TOB = cms.int32(0),
            TID = cms.int32(0),
            TEC = cms.int32(0),
            TIB = cms.int32(0)
        ),
        matchedRecHitsUseAnd = cms.bool(True),
        checkCharge = cms.bool(False)
    ),
    PositiveYOnly = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    writeTriplets = cms.bool(False),
    maxSeeds = cms.int32(20000),
    rescaleError = cms.double(1.0)
)


process.simpleCosmicBONSeedsTop = cms.EDProducer("SimpleCosmicBONSeeder",
    TripletsSrc = cms.InputTag("simpleCosmicBONSeedingLayersTop"),
    helixDebugLevel = cms.untracked.uint32(0),
    seedOnMiddle = cms.bool(False),
    RegionPSet = cms.PSet(
        pMin = cms.double(1.0),
        originRadius = cms.double(150.0),
        ptMin = cms.double(0.5),
        originZPosition = cms.double(0.0),
        originHalfLength = cms.double(90.0)
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(300),
        DontCountDetsAboveNClusters = cms.uint32(20),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(150),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClustersTop")
    ),
    HitsPerModuleCheck = cms.PSet(
        checkHitsPerModule = cms.bool(True),
        Thresholds = cms.PSet(
            TOB = cms.int32(20),
            TID = cms.int32(20),
            TEC = cms.int32(20),
            TIB = cms.int32(20)
        )
    ),
    minimumGoodHitsInSeed = cms.int32(3),
    seedDebugLevel = cms.untracked.uint32(0),
    TripletsDebugLevel = cms.untracked.uint32(0),
    NegativeYOnly = cms.bool(False),
    maxTriplets = cms.int32(50000),
    ClusterChargeCheck = cms.PSet(
        Thresholds = cms.PSet(
            TOB = cms.int32(0),
            TID = cms.int32(0),
            TEC = cms.int32(0),
            TIB = cms.int32(0)
        ),
        matchedRecHitsUseAnd = cms.bool(True),
        checkCharge = cms.bool(False)
    ),
    PositiveYOnly = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    writeTriplets = cms.bool(False),
    maxSeeds = cms.int32(20000),
    rescaleError = cms.double(1.0)
)


process.splittedTracksP5 = cms.EDProducer("TrackProducer",
    src = cms.InputTag("cosmicTrackSplitter"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('RKFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    GeometricInnerState = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('cosmic'),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.stripPairElectronSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    TID = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        skipClusters = cms.InputTag("tripletElectronClusterMask"),
        maxRing = cms.int32(2)
    ),
    layerList = cms.vstring('TIB1+TIB2', 
        'TIB1+TID1_pos', 
        'TIB1+TID1_neg', 
        'TID2_pos+TID3_pos', 
        'TID2_neg+TID3_neg', 
        'TEC1_pos+TEC2_pos', 
        'TEC2_pos+TEC3_pos', 
        'TEC3_pos+TEC4_pos', 
        'TEC3_pos+TEC5_pos', 
        'TEC1_neg+TEC2_neg', 
        'TEC2_neg+TEC3_neg', 
        'TEC3_neg+TEC4_neg', 
        'TEC3_neg+TEC5_neg'),
    TEC = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        skipClusters = cms.InputTag("tripletElectronClusterMask"),
        maxRing = cms.int32(2)
    ),
    TIB = cms.PSet(
        skipClusters = cms.InputTag("tripletElectronClusterMask"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)


process.stripPairElectronSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        maxElement = cms.uint32(1000000),
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.InputTag("stripPairElectronSeedLayers")
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            originRadius = cms.double(0.4),
            ptMin = cms.double(1.0),
            originHalfLength = cms.double(12.0)
        ),
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.tobTecStepClusters = cms.EDProducer("TrackClusterRemover",
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
    trajectories = cms.InputTag("pixelLessStepTracks"),
    oldClusterRemovalInfo = cms.InputTag("pixelLessStepClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag("pixelLessStep"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.tobTecStepSeedLayersPair = cms.EDProducer("SeedingLayersEDProducer",
    TOB = cms.PSet(
        skipClusters = cms.InputTag("tobTecStepClusters"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    layerList = cms.vstring('TOB1+TEC1_pos', 
        'TOB1+TEC1_neg', 
        'TEC1_pos+TEC2_pos', 
        'TEC1_neg+TEC2_neg', 
        'TEC2_pos+TEC3_pos', 
        'TEC2_neg+TEC3_neg', 
        'TEC3_pos+TEC4_pos', 
        'TEC3_neg+TEC4_neg', 
        'TEC4_pos+TEC5_pos', 
        'TEC4_neg+TEC5_neg', 
        'TEC5_pos+TEC6_pos', 
        'TEC5_neg+TEC6_neg', 
        'TEC6_pos+TEC7_pos', 
        'TEC6_neg+TEC7_neg'),
    TEC = cms.PSet(
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        skipClusters = cms.InputTag("tobTecStepClusters"),
        maxRing = cms.int32(5)
    )
)


process.tobTecStepSeedLayersTripl = cms.EDProducer("SeedingLayersEDProducer",
    TOB = cms.PSet(
        skipClusters = cms.InputTag("tobTecStepClusters"),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    layerList = cms.vstring('TOB1+TOB2+MTOB3', 
        'TOB1+TOB2+MTEC1_pos', 
        'TOB1+TOB2+MTEC1_neg'),
    MTEC = cms.PSet(
        minRing = cms.int32(6),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        skipClusters = cms.InputTag("tobTecStepClusters"),
        maxRing = cms.int32(7)
    ),
    MTOB = cms.PSet(
        skipClusters = cms.InputTag("tobTecStepClusters"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    )
)


process.tobTecStepSeeds = cms.EDProducer("SeedCombiner",
    seedCollections = cms.VInputTag(cms.InputTag("tobTecStepSeedsTripl"), cms.InputTag("tobTecStepSeedsPair"))
)


process.tobTecStepSeedsPair = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        maxElement = cms.uint32(1000000),
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.InputTag("tobTecStepSeedLayersPair")
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterStripHits = cms.bool(True),
        ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
        ClusterShapeHitFilterName = cms.string('tobTecStepClusterShapeHitFilter'),
        FilterPixelHits = cms.bool(False),
        FilterAtHelixStage = cms.bool(True)
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            originRadius = cms.double(6.0),
            ptMin = cms.double(0.6),
            originHalfLength = cms.double(30.0)
        ),
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.tobTecStepSeedsTripl = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardMultiHitGenerator'),
        GeneratorPSet = cms.PSet(
            detIdsToDebug = cms.vint32(0, 0, 0),
            pt_interv = cms.vdouble(0.4, 0.7, 1.0, 2.0),
            useFixedPreFiltering = cms.bool(False),
            refitHits = cms.bool(True),
            chi2VsPtCut = cms.bool(True),
            maxChi2 = cms.double(5.0),
            extraHitRPhitolerance = cms.double(0.0),
            extraRKDBox = cms.double(0.2),
            chi2_cuts = cms.vdouble(3.0, 4.0, 5.0, 5.0),
            extraZKDBox = cms.double(0.2),
            extraPhiKDBox = cms.double(0.01),
            maxElement = cms.uint32(100000),
            TTRHBuilder = cms.string('WithTrackAngle'),
            phiPreFiltering = cms.double(0.3),
            debug = cms.bool(False),
            extraHitRZtolerance = cms.double(0.0),
            ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
            ComponentName = cms.string('MultiHitGeneratorFromChi2'),
            fnSigmaRZ = cms.double(2.0)
        ),
        SeedingLayers = cms.InputTag("tobTecStepSeedLayersTripl")
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterStripHits = cms.bool(True),
        ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
        ClusterShapeHitFilterName = cms.string('tobTecStepClusterShapeHitFilter'),
        FilterPixelHits = cms.bool(False),
        FilterAtHelixStage = cms.bool(True)
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            originRadius = cms.double(3.5),
            ptMin = cms.double(0.55),
            originHalfLength = cms.double(20.0)
        ),
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.tobTecStepSelector = cms.EDProducer("MultiTrackSelector",
    src = cms.InputTag("tobTecStepTracks"),
    trackSelectors = cms.VPSet(cms.PSet(
        max_d0 = cms.double(100.0),
        minNumber3DLayers = cms.uint32(0),
        max_lostHitFraction = cms.double(1.0),
        applyAbsCutsIfNoPV = cms.bool(False),
        qualityBit = cms.string('loose'),
        minNumberLayers = cms.uint32(0),
        chi2n_par = cms.double(9999),
        nSigmaZ = cms.double(4.0),
        dz_par2 = cms.vdouble(1.8, 4.0),
        applyAdaptedPVCuts = cms.bool(True),
        min_eta = cms.double(-9999.0),
        dz_par1 = cms.vdouble(1.8, 4.0),
        copyTrajectories = cms.untracked.bool(False),
        vtxNumber = cms.int32(-1),
        keepAllTracks = cms.bool(False),
        maxNumberLostLayers = cms.uint32(999),
        max_relpterr = cms.double(9999.0),
        copyExtras = cms.untracked.bool(True),
        minMVA = cms.double(-0.6),
        vertexCut = cms.string('ndof>=2&!isFake'),
        max_z0 = cms.double(100.0),
        min_nhits = cms.uint32(0),
        name = cms.string('tobTecStepLoose'),
        max_minMissHitOutOrIn = cms.int32(99),
        chi2n_no1Dmod_par = cms.double(9999),
        res_par = cms.vdouble(0.003, 0.01),
        useMVA = cms.bool(True),
        max_eta = cms.double(9999.0),
        d0_par2 = cms.vdouble(2.0, 4.0),
        d0_par1 = cms.vdouble(2.0, 4.0),
        preFilterName = cms.string(''),
        minHitsToBypassChecks = cms.uint32(20)
    ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(2),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('tight'),
            minNumberLayers = cms.uint32(5),
            chi2n_par = cms.double(0.3),
            dz_par1 = cms.vdouble(1.4, 4.0),
            dz_par2 = cms.vdouble(1.4, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            nSigmaZ = cms.double(4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(0),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('tobTecStepTight'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            preFilterName = cms.string('tobTecStepLoose'),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.5, 4.0),
            d0_par1 = cms.vdouble(1.5, 4.0),
            res_par = cms.vdouble(0.003, 0.001),
            minHitsToBypassChecks = cms.uint32(20)
        ), 
        cms.PSet(
            max_d0 = cms.double(100.0),
            minNumber3DLayers = cms.uint32(0),
            max_lostHitFraction = cms.double(1.0),
            applyAbsCutsIfNoPV = cms.bool(False),
            qualityBit = cms.string('highPurity'),
            minNumberLayers = cms.uint32(0),
            chi2n_par = cms.double(9999),
            nSigmaZ = cms.double(4.0),
            dz_par2 = cms.vdouble(1.1, 4.0),
            applyAdaptedPVCuts = cms.bool(True),
            min_eta = cms.double(-9999.0),
            dz_par1 = cms.vdouble(1.1, 4.0),
            copyTrajectories = cms.untracked.bool(False),
            vtxNumber = cms.int32(-1),
            keepAllTracks = cms.bool(True),
            maxNumberLostLayers = cms.uint32(999),
            max_relpterr = cms.double(9999.0),
            copyExtras = cms.untracked.bool(True),
            minMVA = cms.double(0.6),
            vertexCut = cms.string('ndof>=2&!isFake'),
            max_z0 = cms.double(100.0),
            min_nhits = cms.uint32(0),
            name = cms.string('tobTecStep'),
            max_minMissHitOutOrIn = cms.int32(99),
            chi2n_no1Dmod_par = cms.double(9999),
            res_par = cms.vdouble(0.003, 0.01),
            useMVA = cms.bool(True),
            max_eta = cms.double(9999.0),
            d0_par2 = cms.vdouble(1.2, 4.0),
            d0_par1 = cms.vdouble(1.2, 4.0),
            preFilterName = cms.string('tobTecStepLoose'),
            minHitsToBypassChecks = cms.uint32(20)
        )),
    GBRForestLabel = cms.string('MVASelectorIter6_13TeV_v0'),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vertices = cms.InputTag("pixelVertices"),
    useVtxError = cms.bool(False),
    useAnyMVA = cms.bool(True),
    useVertices = cms.bool(True)
)


process.tobTecStepTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    src = cms.InputTag("tobTecStepSeeds"),
    maxSeedsBeforeCleaning = cms.uint32(5000),
    SimpleMagneticField = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        numberMeasurementsForFit = cms.int32(4),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrajectoryCleaner = cms.string('tobTecStepTrajectoryCleanerBySharedHits'),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    cleanTrajectoryAfterInOut = cms.bool(True),
    useHitsSplitting = cms.bool(True),
    numHitsForSeedCleaner = cms.int32(50),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(500000),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('tobTecStepTrajectoryBuilder')
    ),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    clustersToSkip = cms.InputTag("tobTecStepClusters")
)


process.tobTecStepTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("tobTecStepTrackCandidates"),
    SimpleMagneticField = cms.string(''),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    Fitter = cms.string('tobTecFlexibleKFFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    MeasurementTracker = cms.string(''),
    AlgorithmName = cms.string('iter6'),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    GeometricInnerState = cms.bool(False),
    useSimpleMF = cms.bool(False),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


process.topBottomClusterInfoProducer = cms.EDProducer("TopBottomClusterInfoProducer",
    pixelClustersNew = cms.InputTag("siPixelClustersTop"),
    stripStereoHitsNew = cms.InputTag("siStripMatchedRecHitsTop","stereoRecHit"),
    stripMonoHitsNew = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit"),
    stripClustersOld = cms.InputTag("siStripClusters"),
    pixelHitsNew = cms.InputTag("siPixelRecHitsTop"),
    pixelHitsOld = cms.InputTag("siPixelRecHits"),
    pixelClustersOld = cms.InputTag("siPixelClusters"),
    stripClustersNew = cms.InputTag("siStripClustersTop"),
    stripMonoHitsOld = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    stripStereoHitsOld = cms.InputTag("siStripMatchedRecHits","stereoRecHit")
)


process.topBottomClusterInfoProducerBottom = cms.EDProducer("TopBottomClusterInfoProducer",
    stripStereoHitsNew = cms.InputTag("siStripMatchedRecHitsBottom","stereoRecHit"),
    stripMonoHitsNew = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit"),
    stripClustersOld = cms.InputTag("siStripClusters"),
    pixelHitsNew = cms.InputTag("siPixelRecHitsBottom"),
    pixelHitsOld = cms.InputTag("siPixelRecHits"),
    stripStereoHitsOld = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    pixelClustersOld = cms.InputTag("siPixelClusters"),
    stripClustersNew = cms.InputTag("siStripClustersBottom"),
    stripMonoHitsOld = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    pixelClustersNew = cms.InputTag("siPixelClustersBottom")
)


process.topBottomClusterInfoProducerTop = cms.EDProducer("TopBottomClusterInfoProducer",
    stripStereoHitsNew = cms.InputTag("siStripMatchedRecHitsTop","stereoRecHit"),
    stripMonoHitsNew = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit"),
    stripClustersOld = cms.InputTag("siStripClusters"),
    pixelHitsNew = cms.InputTag("siPixelRecHitsTop"),
    pixelHitsOld = cms.InputTag("siPixelRecHits"),
    stripStereoHitsOld = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    pixelClustersOld = cms.InputTag("siPixelClusters"),
    stripClustersNew = cms.InputTag("siStripClustersTop"),
    stripMonoHitsOld = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    pixelClustersNew = cms.InputTag("siPixelClustersTop")
)


process.trackExtrapolator = cms.EDProducer("TrackExtrapolator",
    trackQuality = cms.string('goodIterative'),
    trackSrc = cms.InputTag("generalTracks")
)


process.trackRefsForJets = cms.EDProducer("ChargedRefCandidateProducer",
    src = cms.InputTag("trackWithVertexRefSelector"),
    particleType = cms.string('pi+')
)


process.tripletElectronClusterMask = cms.EDProducer("SeedClusterRemover",
    trajectories = cms.InputTag("tripletElectronSeeds"),
    oldClusterRemovalInfo = cms.InputTag("pixelLessStepSeedClusterMask"),
    stripClusters = cms.InputTag("siStripClusters"),
    overrideTrkQuals = cms.InputTag(""),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    ),
    TrackQuality = cms.string('highPurity'),
    clusterLessSolution = cms.bool(True)
)


process.tripletElectronSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    FPix = cms.PSet(
        skipClusters = cms.InputTag("pixelLessStepSeedClusterMask"),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        skipClusters = cms.InputTag("pixelLessStepSeedClusterMask"),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('siPixelRecHits')
    )
)


process.tripletElectronSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        GeneratorPSet = cms.PSet(
            useBending = cms.bool(True),
            useFixedPreFiltering = cms.bool(False),
            maxElement = cms.uint32(1000000),
            SeedComparitorPSet = cms.PSet(
                ComponentName = cms.string('none')
            ),
            extraHitRPhitolerance = cms.double(0.032),
            useMultScattering = cms.bool(True),
            phiPreFiltering = cms.double(0.3),
            extraHitRZtolerance = cms.double(0.037),
            ComponentName = cms.string('PixelTripletHLTGenerator')
        ),
        SeedingLayers = cms.InputTag("tripletElectronSeedLayers")
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    ClusterCheckPSet = cms.PSet(
        MaxNumberOfPixelClusters = cms.uint32(40000),
        cut = cms.string('strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)'),
        PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
        MaxNumberOfCosmicClusters = cms.uint32(400000),
        doClusterCheck = cms.bool(True),
        ClusterCollectionLabel = cms.InputTag("siStripClusters")
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originRadius = cms.double(0.02),
            nSigmaZ = cms.double(4.0),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            ptMin = cms.double(1.0)
        )
    ),
    SeedCreatorPSet = cms.PSet(
        SimpleMagneticField = cms.string('ParabolicMf'),
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        SeedMomentumForBOFF = cms.double(5.0),
        OriginTransverseErrorMultiplier = cms.double(1.0),
        TTRHBuilder = cms.string('WithTrackAngle'),
        MinOneOverPtError = cms.double(1.0),
        propagator = cms.string('PropagatorWithMaterial')
    )
)


process.HighPuritySelector = cms.EDFilter("AlignmentTrackSelectorModule",
    theCharge = cms.int32(0),
    minHitChargeStrip = cms.double(20.0),
    minHitsPerSubDet = cms.PSet(
        inPIXEL = cms.int32(0),
        inENDCAPminus = cms.int32(0),
        inFPIXminus = cms.int32(0),
        inTIDplus = cms.int32(0),
        inTECminus = cms.int32(0),
        inFPIXplus = cms.int32(0),
        inTOB = cms.int32(0),
        inTEC = cms.int32(0),
        inTECplus = cms.int32(0),
        inENDCAPplus = cms.int32(0),
        inTID = cms.int32(0),
        inFPIX = cms.int32(0),
        inENDCAP = cms.int32(0),
        inBPIX = cms.int32(0),
        inTIDminus = cms.int32(0),
        inTIB = cms.int32(0)
    ),
    applyMultiplicityFilter = cms.bool(False),
    matchedrecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    countStereoHitAs2D = cms.bool(True),
    pMin = cms.double(0.0),
    RorZofLastHitMax = cms.vdouble(999.0, 999.0),
    etaMin = cms.double(-999.0),
    dzMax = cms.double(999999.0),
    etaMax = cms.double(999.0),
    pMax = cms.double(9999.0),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    GlobalSelector = cms.PSet(
        applyIsolationtest = cms.bool(False),
        muonSource = cms.InputTag("muons"),
        minIsolatedCount = cms.int32(0),
        jetIsoSource = cms.InputTag("kt6CaloJets"),
        minGlobalMuonCount = cms.int32(1),
        minJetDeltaR = cms.double(0.2),
        maxJetCount = cms.int32(3),
        applyGlobalMuonFilter = cms.bool(False),
        minJetPt = cms.double(40.0),
        jetCountSource = cms.InputTag("kt6CaloJets"),
        maxJetPt = cms.double(40.0),
        applyJetCountFilter = cms.bool(False),
        maxTrackDeltaR = cms.double(0.001)
    ),
    maxHitDiffEndcaps = cms.double(999.0),
    dzMin = cms.double(-999999.0),
    TwoBodyDecaySelector = cms.PSet(
        applyMassrangeFilter = cms.bool(False),
        daughterMass = cms.double(0.105),
        useUnsignedCharge = cms.bool(True),
        missingETSource = cms.InputTag("met"),
        maxXMass = cms.double(15000.0),
        numberOfCandidates = cms.uint32(1),
        charge = cms.int32(0),
        acoplanarDistance = cms.double(1.0),
        minXMass = cms.double(0.0),
        applySecThreshold = cms.bool(False),
        applyChargeFilter = cms.bool(False),
        applyAcoplanarityFilter = cms.bool(False),
        secondThreshold = cms.double(6.0),
        applyMissingETFilter = cms.bool(False)
    ),
    minPrescaledHits = cms.int32(-1),
    minHitIsolation = cms.double(0.01),
    minMultiplicity = cms.int32(1),
    nHitMin = cms.double(0.0),
    ptMax = cms.double(999.0),
    ptMin = cms.double(0.0),
    nHitMax = cms.double(999.0),
    d0Max = cms.double(999999.0),
    iterativeTrackingSteps = cms.vstring(),
    trackQualities = cms.vstring('highPurity'),
    applyNHighestPt = cms.bool(False),
    d0Min = cms.double(-999999.0),
    RorZofFirstHitMin = cms.vdouble(0.0, 0.0),
    nLostHitMax = cms.double(999.0),
    chi2nMax = cms.double(999999.0),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    hitPrescaleMapTag = cms.InputTag(""),
    nHighestPt = cms.int32(2),
    nHitMin2D = cms.uint32(0),
    src = cms.InputTag("MuSkim"),
    applyIsolationCut = cms.bool(False),
    multiplicityOnInput = cms.bool(False),
    RorZofFirstHitMax = cms.vdouble(999.0, 999.0),
    RorZofLastHitMin = cms.vdouble(0.0, 0.0),
    filter = cms.bool(True),
    maxMultiplicity = cms.int32(999999),
    seedOnlyFrom = cms.int32(0),
    applyChargeCheck = cms.bool(False),
    applyBasicCuts = cms.bool(True)
)


process.TriggerFilter = cms.EDFilter("HLTHighLevel",
    eventSetupPathsKey = cms.string(''),
    andOr = cms.bool(True),
    HLTPaths = cms.vstring('HLT_IsoMu24_*'),
    throw = cms.bool(False),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


process.firstStepGoodPrimaryVertices = cms.EDFilter("PrimaryVertexObjectFilter",
    src = cms.InputTag("firstStepPrimaryVertices"),
    filterParams = cms.PSet(
        maxZ = cms.double(15.0),
        minNdof = cms.double(25.0),
        maxRho = cms.double(2.0)
    )
)


process.jetsForCoreTracking = cms.EDFilter("CandPtrSelector",
    src = cms.InputTag("ak4CaloJetsForTrk"),
    cut = cms.string('pt > 100 && abs(eta) < 2.5')
)


process.ApeEstimator1 = cms.EDAnalyzer("ApeEstimator",
    maxTracksPerEvent = cms.uint32(0),
    TrackerTreeFile = cms.string('/afs/cern.ch/work/a/ajkumar/APE_newCPE_v1/CMSSW_7_2_0_pre6/src/Alignment/TrackerTreeGenerator/hists/TrackerTree.root'),
    applyTrackCuts = cms.bool(True),
    Sectors = cms.VPSet(cms.PSet(
        layer = cms.vuint32(1),
        outerInner = cms.vuint32(),
        petal = cms.vuint32(),
        bladeAl = cms.vuint32(),
        rod = cms.vuint32(),
        module = cms.vuint32(),
        ring = cms.vuint32(),
        isRPhi = cms.vuint32(),
        subdetId = cms.vuint32(1),
        uDirection = cms.vint32(),
        wDirection = cms.vint32(1),
        nStrips = cms.vuint32(),
        isDoubleSide = cms.vuint32(),
        vDirection = cms.vint32(),
        posEta = cms.vdouble(),
        half = cms.vuint32(),
        rawId = cms.vuint32(),
        panel = cms.vuint32(),
        name = cms.string('BpixLayer1Out'),
        blade = cms.vuint32(),
        rodAl = cms.vuint32(),
        posPhi = cms.vdouble(),
        posZ = cms.vdouble(),
        posX = cms.vdouble(),
        posY = cms.vdouble(),
        posR = cms.vdouble(),
        side = cms.vuint32()
    ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(1),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('BpixLayer1In'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(2),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(1),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('BpixLayer2Out'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(2),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(1),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('BpixLayer2In'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(3),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(1),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('BpixLayer3Out'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(3),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(1),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('BpixLayer3In'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(2),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('FpixMinusLayer1'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(1)
        ), 
        cms.PSet(
            layer = cms.vuint32(2),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(2),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('FpixMinusLayer2'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(1)
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(2),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('FpixPlusLayer1'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(2)
        ), 
        cms.PSet(
            layer = cms.vuint32(2),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(2),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('FpixPlusLayer2'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(2)
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer1RphiOut'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer1StereoOut'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer1RphiIn'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer1StereoIn'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(2),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer2RphiOut'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(2),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer2StereoOut'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(2),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer2RphiIn'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(2),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer2StereoIn'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(3),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer3Out'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(3),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer3In'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(4),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer4Out'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(4),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer4In'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer1StereoOut'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer1RphiIn'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(2),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer2RphiOut'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(2),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer2StereoIn'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(3),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer3Out'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(3),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer3In'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(4),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer4Out'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(4),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer4In'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(5),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer5Out'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(5),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer5In'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(6),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer6Out'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(6),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer6In'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(1),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(4),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TidMinusRing1Rphi'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(1),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(1),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(4),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TidMinusRing1Stereo'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(1),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(2),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(4),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TidMinusRing2Rphi'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(1),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(2),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(4),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TidMinusRing2Stereo'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(1),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(3),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(4),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TidMinusRing3'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(1)
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(1),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(4),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TidPlusRing1Rphi'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(2),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(1),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(4),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TidPlusRing1Stereo'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(2),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(2),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(4),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TidPlusRing2Rphi'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(2),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(2),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(4),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TidPlusRing2Stereo'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(2),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(3),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(4),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TidPlusRing3'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(2)
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(1),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecMinusRing1Rphi'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(1),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(1),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecMinusRing1Stereo'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(1),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(2),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecMinusRing2Rphi'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(1),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(2),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecMinusRing2Stereo'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(1),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(3),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecMinusRing3'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(1)
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(4),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecMinusRing4'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(1)
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(5),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecMinusRing5Rphi'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(1),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(5),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecMinusRing5Stereo'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(1),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(6),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecMinusRing6'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(1)
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(7),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecMinusRing7'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(1)
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(1),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecPlusRing1Rphi'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(2),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(1),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecPlusRing1Stereo'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(2),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(2),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecPlusRing2Rphi'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(2),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(2),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecPlusRing2Stereo'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(2),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(3),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecPlusRing3'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(2)
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(4),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecPlusRing4'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(2)
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(5),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecPlusRing5Rphi'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(2),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(5),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecPlusRing5Stereo'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(2),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(6),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecPlusRing6'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(2)
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(7),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecPlusRing7'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(2)
        )),
    residualErrorBinning = cms.vdouble(0.0005, 0.001, 0.0015, 0.002, 0.0025, 
        0.003, 0.0035, 0.004, 0.005, 0.007, 
        0.01),
    tjTkAssociationMapTag = cms.InputTag("TrackRefitterHighPurityForApeEstimator"),
    zoomHists = cms.bool(True),
    calculateApe = cms.bool(False),
    vErrHists = cms.vuint32(1),
    HitSelector = cms.PSet(
        phiSens = cms.vdouble(),
        clusterProbabilityXY = cms.vdouble(),
        chargeLRplus = cms.vdouble(),
        chargePixel = cms.vdouble(10000.0, 2000000.0),
        sOverN = cms.vdouble(20.0, 50.0),
        errY2 = cms.vdouble(),
        clusterProbabilityXYQ = cms.vdouble(),
        widthDiff = cms.vdouble(),
        resX = cms.vdouble(),
        resY = cms.vdouble(),
        widthProj = cms.vdouble(),
        chargeAsymmetry = cms.vdouble(),
        chargeOnEdges = cms.vdouble(0.0, 0.5),
        isOnEdge = cms.vuint32(0, 0),
        width = cms.vuint32(3, 3),
        errY = cms.vdouble(),
        errX = cms.vdouble(),
        probX = cms.vdouble(),
        probY = cms.vdouble(),
        norResX = cms.vdouble(),
        norResY = cms.vdouble(),
        widthX = cms.vuint32(2, 1000),
        widthY = cms.vuint32(2, 1000),
        errYHit = cms.vdouble(),
        clusterProbabilityQ = cms.vdouble(),
        charge = cms.vdouble(),
        spansTwoRoc = cms.vuint32(),
        errXTrk = cms.vdouble(),
        phiSensX = cms.vdouble(),
        baryStripX = cms.vdouble(),
        errX2 = cms.vdouble(),
        maxCharge = cms.vdouble(),
        baryStripY = cms.vdouble(),
        errYTrk = cms.vdouble(),
        qBin = cms.vuint32(1, 3),
        hasBadPixels = cms.vuint32(),
        phiSensY = cms.vdouble(),
        maxIndex = cms.vuint32(1, 1),
        edgeStrips = cms.vuint32(),
        errXHit = cms.vdouble(),
        chargeLRminus = cms.vdouble(),
        logClusterProbability = cms.vdouble(-5.0, 1.0)
    ),
    minGoodHitsPerTrack = cms.uint32(1),
    analyzerMode = cms.bool(False)
)


process.ApeEstimator2 = cms.EDAnalyzer("ApeEstimator",
    maxTracksPerEvent = cms.uint32(0),
    TrackerTreeFile = cms.string('/afs/cern.ch/work/a/ajkumar/APE_newCPE_v1/CMSSW_7_2_0_pre6/src/Alignment/TrackerTreeGenerator/hists/TrackerTree.root'),
    applyTrackCuts = cms.bool(True),
    Sectors = cms.VPSet(cms.PSet(
        layer = cms.vuint32(1),
        outerInner = cms.vuint32(),
        petal = cms.vuint32(),
        bladeAl = cms.vuint32(),
        rod = cms.vuint32(),
        module = cms.vuint32(),
        ring = cms.vuint32(),
        isRPhi = cms.vuint32(),
        subdetId = cms.vuint32(1),
        uDirection = cms.vint32(),
        wDirection = cms.vint32(1),
        nStrips = cms.vuint32(),
        isDoubleSide = cms.vuint32(),
        vDirection = cms.vint32(),
        posEta = cms.vdouble(),
        half = cms.vuint32(),
        rawId = cms.vuint32(),
        panel = cms.vuint32(),
        name = cms.string('BpixLayer1Out'),
        blade = cms.vuint32(),
        rodAl = cms.vuint32(),
        posPhi = cms.vdouble(),
        posZ = cms.vdouble(),
        posX = cms.vdouble(),
        posY = cms.vdouble(),
        posR = cms.vdouble(),
        side = cms.vuint32()
    ), 
        cms.PSet(
            layer = cms.vuint32(3),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(1),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('BpixLayer3In'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(2),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('FpixMinusLayer1'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(1)
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer1RphiOut'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(4),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer4In'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer1StereoOut'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(5),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer5Out'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(7),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecPlusRing7'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(2)
        )),
    residualErrorBinning = cms.vdouble(0.0005, 0.001, 0.0015, 0.002, 0.0025, 
        0.003, 0.0035, 0.004, 0.005, 0.007, 
        0.01),
    tjTkAssociationMapTag = cms.InputTag("TrackRefitterHighPurityForApeEstimator"),
    zoomHists = cms.bool(True),
    calculateApe = cms.bool(False),
    vErrHists = cms.vuint32(1),
    HitSelector = cms.PSet(
        phiSens = cms.vdouble(),
        clusterProbabilityXY = cms.vdouble(),
        chargeLRplus = cms.vdouble(),
        chargePixel = cms.vdouble(10000.0, 2000000.0),
        sOverN = cms.vdouble(20.0, 50.0),
        errY2 = cms.vdouble(),
        clusterProbabilityXYQ = cms.vdouble(),
        widthDiff = cms.vdouble(),
        resX = cms.vdouble(),
        resY = cms.vdouble(),
        widthProj = cms.vdouble(),
        chargeAsymmetry = cms.vdouble(),
        chargeOnEdges = cms.vdouble(0.0, 0.5),
        isOnEdge = cms.vuint32(0, 0),
        width = cms.vuint32(3, 3),
        errY = cms.vdouble(),
        errX = cms.vdouble(),
        probX = cms.vdouble(),
        probY = cms.vdouble(),
        norResX = cms.vdouble(),
        norResY = cms.vdouble(),
        widthX = cms.vuint32(2, 1000),
        widthY = cms.vuint32(2, 1000),
        errYHit = cms.vdouble(),
        clusterProbabilityQ = cms.vdouble(),
        charge = cms.vdouble(),
        spansTwoRoc = cms.vuint32(),
        errXTrk = cms.vdouble(),
        phiSensX = cms.vdouble(),
        baryStripX = cms.vdouble(),
        errX2 = cms.vdouble(),
        maxCharge = cms.vdouble(),
        baryStripY = cms.vdouble(),
        errYTrk = cms.vdouble(),
        qBin = cms.vuint32(1, 3),
        hasBadPixels = cms.vuint32(),
        phiSensY = cms.vdouble(),
        maxIndex = cms.vuint32(1, 1),
        edgeStrips = cms.vuint32(),
        errXHit = cms.vdouble(),
        chargeLRminus = cms.vdouble(),
        logClusterProbability = cms.vdouble(-5.0, 1.0)
    ),
    minGoodHitsPerTrack = cms.uint32(1),
    analyzerMode = cms.bool(True)
)


process.ApeEstimator3 = cms.EDAnalyzer("ApeEstimator",
    maxTracksPerEvent = cms.uint32(0),
    TrackerTreeFile = cms.string('/afs/cern.ch/work/a/ajkumar/APE_newCPE_v1/CMSSW_7_2_0_pre6/src/Alignment/TrackerTreeGenerator/hists/TrackerTree.root'),
    applyTrackCuts = cms.bool(True),
    Sectors = cms.VPSet(cms.PSet(
        layer = cms.vuint32(1),
        outerInner = cms.vuint32(),
        petal = cms.vuint32(),
        bladeAl = cms.vuint32(),
        rod = cms.vuint32(),
        module = cms.vuint32(),
        ring = cms.vuint32(),
        isRPhi = cms.vuint32(),
        subdetId = cms.vuint32(1),
        uDirection = cms.vint32(),
        wDirection = cms.vint32(1),
        nStrips = cms.vuint32(),
        isDoubleSide = cms.vuint32(),
        vDirection = cms.vint32(),
        posEta = cms.vdouble(),
        half = cms.vuint32(),
        rawId = cms.vuint32(),
        panel = cms.vuint32(),
        name = cms.string('BpixLayer1Out'),
        blade = cms.vuint32(),
        rodAl = cms.vuint32(),
        posPhi = cms.vdouble(),
        posZ = cms.vdouble(),
        posX = cms.vdouble(),
        posY = cms.vdouble(),
        posR = cms.vdouble(),
        side = cms.vuint32()
    ), 
        cms.PSet(
            layer = cms.vuint32(3),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(1),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('BpixLayer3In'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(2),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('FpixMinusLayer1'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(1)
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(1),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer1RphiOut'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(4),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(3),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(-1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TibLayer4In'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(1),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(2),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            posY = cms.vdouble(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer1StereoOut'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            side = cms.vuint32(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            vDirection = cms.vint32(),
            posR = cms.vdouble(),
            posPhi = cms.vdouble()
        ), 
        cms.PSet(
            layer = cms.vuint32(5),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(5),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(1),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TobLayer5Out'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32()
        ), 
        cms.PSet(
            layer = cms.vuint32(),
            outerInner = cms.vuint32(),
            petal = cms.vuint32(),
            bladeAl = cms.vuint32(),
            rod = cms.vuint32(),
            module = cms.vuint32(),
            ring = cms.vuint32(7),
            isRPhi = cms.vuint32(),
            subdetId = cms.vuint32(6),
            uDirection = cms.vint32(),
            wDirection = cms.vint32(),
            nStrips = cms.vuint32(),
            isDoubleSide = cms.vuint32(),
            vDirection = cms.vint32(),
            posEta = cms.vdouble(),
            half = cms.vuint32(),
            rawId = cms.vuint32(),
            panel = cms.vuint32(),
            name = cms.string('TecPlusRing7'),
            blade = cms.vuint32(),
            rodAl = cms.vuint32(),
            posPhi = cms.vdouble(),
            posZ = cms.vdouble(),
            posX = cms.vdouble(),
            posY = cms.vdouble(),
            posR = cms.vdouble(),
            side = cms.vuint32(2)
        )),
    residualErrorBinning = cms.vdouble(0.0005, 0.001, 0.0015, 0.002, 0.0025, 
        0.003, 0.0035, 0.004, 0.005, 0.007, 
        0.01),
    tjTkAssociationMapTag = cms.InputTag("TrackRefitterHighPurityForApeEstimator"),
    zoomHists = cms.bool(False),
    calculateApe = cms.bool(False),
    vErrHists = cms.vuint32(1),
    HitSelector = cms.PSet(
        phiSens = cms.vdouble(),
        clusterProbabilityXY = cms.vdouble(),
        chargeLRplus = cms.vdouble(),
        chargePixel = cms.vdouble(10000.0, 2000000.0),
        sOverN = cms.vdouble(20.0, 50.0),
        errY2 = cms.vdouble(),
        clusterProbabilityXYQ = cms.vdouble(),
        widthDiff = cms.vdouble(),
        resX = cms.vdouble(),
        resY = cms.vdouble(),
        widthProj = cms.vdouble(),
        chargeAsymmetry = cms.vdouble(),
        chargeOnEdges = cms.vdouble(0.0, 0.5),
        isOnEdge = cms.vuint32(0, 0),
        width = cms.vuint32(3, 3),
        errY = cms.vdouble(),
        errX = cms.vdouble(),
        probX = cms.vdouble(),
        probY = cms.vdouble(),
        norResX = cms.vdouble(),
        norResY = cms.vdouble(),
        widthX = cms.vuint32(2, 1000),
        widthY = cms.vuint32(2, 1000),
        errYHit = cms.vdouble(),
        clusterProbabilityQ = cms.vdouble(),
        charge = cms.vdouble(),
        spansTwoRoc = cms.vuint32(),
        errXTrk = cms.vdouble(),
        phiSensX = cms.vdouble(),
        baryStripX = cms.vdouble(),
        errX2 = cms.vdouble(),
        maxCharge = cms.vdouble(),
        baryStripY = cms.vdouble(),
        errYTrk = cms.vdouble(),
        qBin = cms.vuint32(1, 3),
        hasBadPixels = cms.vuint32(),
        phiSensY = cms.vdouble(),
        maxIndex = cms.vuint32(1, 1),
        edgeStrips = cms.vuint32(),
        errXHit = cms.vdouble(),
        chargeLRminus = cms.vdouble(),
        logClusterProbability = cms.vdouble(-5.0, 1.0)
    ),
    minGoodHitsPerTrack = cms.uint32(1),
    analyzerMode = cms.bool(True)
)


process.regionalCosmicTracksSeq = cms.Sequence(process.regionalCosmicTrackerSeedingLayers+process.regionalCosmicTrackerSeeds+process.regionalCosmicCkfTrackCandidates+process.regionalCosmicTracks)


process.doAlldEdXEstimatorsCTF = cms.Sequence(process.dedxTruncated40CTF+process.dedxDiscrimASmiCTF+process.dedxHarmonic2CTF)


process.electronSeedsSeq = cms.Sequence(process.initialStepSeedClusterMask+process.pixelPairStepSeedClusterMask+process.mixedTripletStepSeedClusterMask+process.pixelLessStepSeedClusterMask+process.tripletElectronSeedLayers+process.tripletElectronSeeds+process.tripletElectronClusterMask+process.pixelPairElectronSeedLayers+process.pixelPairElectronSeeds+process.stripPairElectronSeedLayers+process.stripPairElectronSeeds+process.newCombinedSeeds)


process.muonSeededStepCore = cms.Sequence(process.muonSeededSeedsInOut+process.muonSeededTrackCandidatesInOut+process.muonSeededTracksInOut+process.muonSeededSeedsOutIn+process.muonSeededTrackCandidatesOutIn+process.muonSeededTracksOutIn)


process.ctfTracksCombinedSeeds = cms.Sequence(process.MixedLayerPairs+process.globalSeedsFromPairsWithVertices+process.PixelLayerTriplets+process.globalSeedsFromTriplets+process.globalCombinedSeeds+process.ckfTrackCandidatesCombinedSeeds+process.ctfCombinedSeeds)


process.cosmictracksP5Top = cms.Sequence(process.cosmicseedfinderP5Top+process.cosmicCandidateFinderP5Top+process.cosmictrackfinderP5Top)


process.Conv2Step = cms.Sequence(process.conv2Clusters+process.conv2LayerPairs+process.photonConvTrajSeedFromQuadruplets+process.conv2TrackCandidates+process.conv2StepTracks+process.conv2StepSelector)


process.PixelLessStep = cms.Sequence(process.pixelLessStepClusters+process.pixelLessStepSeedLayers+process.pixelLessStepSeeds+process.pixelLessStepTrackCandidates+process.pixelLessStepTracks+process.pixelLessStepSelector+process.pixelLessStep)


process.RefitterSequence = cms.Sequence(process.offlineBeamSpot+process.TrackRefitterForApeEstimator)


process.TobTecStep = cms.Sequence(process.tobTecStepClusters+process.tobTecStepSeedLayersTripl+process.tobTecStepSeedsTripl+process.tobTecStepSeedLayersPair+process.tobTecStepSeedsPair+process.tobTecStepSeeds+process.tobTecStepTrackCandidates+process.tobTecStepTracks+process.tobTecStepSelector)


process.ctfTracksPixelLess = cms.Sequence(process.pixelLessLayerPairs4PixelLessTracking+process.globalPixelLessSeeds+process.ckfTrackCandidatesPixelLess+process.ctfPixelLess)


process.striptrackerlocalreco = cms.Sequence(process.siStripZeroSuppression+process.siStripClusters+process.siStripMatchedRecHits)


process.JetCoreRegionalStep = cms.Sequence(process.iter0TrackRefsForJets+process.caloTowerForTrk+process.ak4CaloJetsForTrk+process.jetsForCoreTracking+process.firstStepPrimaryVertices+process.firstStepGoodPrimaryVertices+process.jetCoreRegionalStepSeedLayers+process.jetCoreRegionalStepSeeds+process.jetCoreRegionalStepTrackCandidates+process.jetCoreRegionalStepTracks+process.jetCoreRegionalStepSelector)


process.ApeEstimatorSequence = cms.Sequence(process.ApeEstimator1+process.ApeEstimator2+process.ApeEstimator3)


process.doAlldEdXEstimatorsCosmicTF = cms.Sequence(process.dedxTruncated40CosmicTF+process.dedxDiscrimASmiCosmicTF+process.dedxHarmonic2CosmicTF)


process.LowPtTripletStep = cms.Sequence(process.lowPtTripletStepClusters+process.lowPtTripletStepSeedLayers+process.lowPtTripletStepSeeds+process.lowPtTripletStepTrackCandidates+process.lowPtTripletStepTracks+process.lowPtTripletStepSelector)


process.ctfTracksNoOverlaps = cms.Sequence(process.ckfTrackCandidatesNoOverlaps+process.ctfNoOverlaps)


process.doAlldEdXEstimatorsCTFP5LHC = cms.Sequence(process.dedxTruncated40CTFP5LHC+process.dedxDiscrimASmiCTFP5LHC+process.dedxHarmonic2CTFP5LHC)


process.generalTracksSequence = cms.Sequence(process.duplicateTrackCandidates+process.mergedDuplicateTracks+process.duplicateTrackSelector+process.generalTracks)


process.doAlldEdXEstimatorsRS = cms.Sequence(process.dedxTruncated40RS+process.dedxDiscrimASmiRS+process.dedxHarmonic2RS)


process.doAllCosmicdEdXEstimators = cms.Sequence(process.doAlldEdXEstimatorsCTF+process.doAlldEdXEstimatorsCosmicTF+process.doAlldEdXEstimatorsCTFP5LHC)


process.muonSeededStepDebug = cms.Sequence(process.muonSeededSeedsOutInAsTracks+process.muonSeededTrackCandidatesOutInAsTracks+process.muonSeededSeedsInOutAsTracks+process.muonSeededTrackCandidatesInOutAsTracks)


process.muonSeededStepExtra = cms.Sequence(process.muonSeededTracksInOutSelector+process.muonSeededTracksOutInSelector)


process.MixedTripletStep = cms.Sequence(process.mixedTripletStepClusters+process.mixedTripletStepSeedLayersA+process.mixedTripletStepSeedsA+process.mixedTripletStepSeedLayersB+process.mixedTripletStepSeedsB+process.mixedTripletStepSeeds+process.mixedTripletStepTrackCandidates+process.mixedTripletStepTracks+process.mixedTripletStepSelector+process.mixedTripletStep)


process.trackerlocalrecoTop = cms.Sequence(process.siPixelClustersTop+process.siPixelRecHitsTop+process.siStripClustersTop+process.siStripMatchedRecHitsTop+process.topBottomClusterInfoProducerTop)


process.ctftracksP5Top = cms.Sequence(process.combinatorialcosmicseedingtripletsP5Top+process.combinatorialcosmicseedingpairsTOBP5Top+process.combinatorialcosmicseedingpairsTECposP5Top+process.combinatorialcosmicseedingpairsTECnegP5Top+process.combinatorialcosmicseedfinderP5Top+process.simpleCosmicBONSeedingLayersTop+process.simpleCosmicBONSeedsTop+process.combinedP5SeedsForCTFTop+process.ckfTrackCandidatesP5Top+process.ctfWithMaterialTracksP5Top)


process.ConvStep = cms.Sequence(process.convClusters+process.convLayerPairs+process.photonConvTrajSeedFromSingleLeg+process.convTrackCandidates+process.convStepTracks+process.convStepSelector)


process.pixeltrackerlocalreco = cms.Sequence(process.siPixelClusters+process.siPixelRecHits)


process.RefitterHighPuritySequence = cms.Sequence(process.offlineBeamSpot+process.HighPuritySelector+process.TrackRefitterHighPurityForApeEstimator)


process.cosmictracksP5 = cms.Sequence(process.cosmicseedfinderP5+process.cosmicCandidateFinderP5+process.cosmictrackfinderCosmics+process.cosmictrackfinderP5+process.cosmicTrackSplitter+process.splittedTracksP5)


process.DetachedTripletStep = cms.Sequence(process.detachedTripletStepClusters+process.detachedTripletStepSeedLayers+process.detachedTripletStepSeeds+process.detachedTripletStepTrackCandidates+process.detachedTripletStepTracks+process.detachedTripletStepSelector+process.detachedTripletStep)


process.ctftracksP5Bottom = cms.Sequence(process.combinatorialcosmicseedingtripletsP5Bottom+process.combinatorialcosmicseedingpairsTOBP5Bottom+process.combinatorialcosmicseedingpairsTECposP5Bottom+process.combinatorialcosmicseedingpairsTECnegP5Bottom+process.combinatorialcosmicseedfinderP5Bottom+process.simpleCosmicBONSeedingLayersBottom+process.simpleCosmicBONSeedsBottom+process.combinedP5SeedsForCTFBottom+process.ckfTrackCandidatesP5Bottom+process.ctfWithMaterialTracksP5Bottom)


process.beamhaloTracksSeq = cms.Sequence(process.beamhaloTrackerSeedingLayers+process.beamhaloTrackerSeeds+process.beamhaloTrackCandidates+process.beamhaloTracks)


process.TriggerSelectionSequence = cms.Sequence(process.TriggerFilter)


process.combinatorialcosmicseedinglayersP5 = cms.Sequence(process.combinatorialcosmicseedingtripletsP5+process.combinatorialcosmicseedingpairsTOBP5+process.combinatorialcosmicseedingpairsTECposP5+process.combinatorialcosmicseedingpairsTECnegP5)


process.PixelPairStep = cms.Sequence(process.pixelPairStepClusters+process.pixelPairStepSeedLayers+process.pixelPairStepSeeds+process.pixelPairStepTrackCandidates+process.pixelPairStepTracks+process.pixelPairStepSelector)


process.cosmictracksP5Bottom = cms.Sequence(process.cosmicseedfinderP5Bottom+process.cosmicCandidateFinderP5Bottom+process.cosmictrackfinderP5Bottom)


process.doAlldEdXEstimators = cms.Sequence(process.dedxTruncated40+process.dedxHarmonic2+process.dedxDiscrimASmi)


process.InitialStep = cms.Sequence(process.initialStepSeedLayers+process.initialStepSeeds+process.initialStepTrackCandidates+process.initialStepTracks+process.initialStepSelector+process.initialStep)


process.trackerlocalrecoBottom = cms.Sequence(process.siPixelClustersBottom+process.siPixelRecHitsBottom+process.siStripClustersBottom+process.siStripMatchedRecHitsBottom+process.topBottomClusterInfoProducerBottom)


process.tracksP5Top = cms.Sequence(process.ctftracksP5Top+process.cosmictracksP5Top)


process.muonSeededStep = cms.Sequence(process.earlyMuons+process.muonSeededStepCore+process.muonSeededStepExtra)


process.trackerlocalreco = cms.Sequence(process.pixeltrackerlocalreco+process.striptrackerlocalreco)


process.iterTracking = cms.Sequence(process.InitialStep+process.DetachedTripletStep+process.LowPtTripletStep+process.PixelPairStep+process.MixedTripletStep+process.PixelLessStep+process.TobTecStep+process.JetCoreRegionalStep+process.earlyGeneralTracks+process.muonSeededStep+process.preDuplicateMergingGeneralTracks+process.generalTracksSequence+process.ConvStep+process.conversionStepTracks)


process.ctftracksP5 = cms.Sequence(process.combinatorialcosmicseedinglayersP5+process.combinatorialcosmicseedfinderP5+process.simpleCosmicBONSeedingLayers+process.simpleCosmicBONSeeds+process.combinedP5SeedsForCTF+process.ckfTrackCandidatesP5+process.ctfWithMaterialTracksCosmics+process.ctfWithMaterialTracksP5+process.ckfTrackCandidatesP5LHCNavigation+process.ctfWithMaterialTracksP5LHCNavigation)


process.tracksP5 = cms.Sequence(process.cosmictracksP5+process.ctftracksP5+process.doAllCosmicdEdXEstimators)


process.tracksP5Bottom = cms.Sequence(process.ctftracksP5Bottom+process.cosmictracksP5Bottom)


process.ckftracks = cms.Sequence(process.iterTracking+process.electronSeedsSeq+process.doAlldEdXEstimators)


process.ckftracks_wodEdX = cms.Sequence(process.iterTracking+process.electronSeedsSeq)


process.trackingGlobalReco = cms.Sequence(process.ckftracks+process.trackExtrapolator)


process.tracksP5_wodEdX = cms.Sequence(process.cosmictracksP5+process.ctftracksP5)


process.ckftracks_woBH = cms.Sequence(process.iterTracking+process.electronSeedsSeq+process.doAlldEdXEstimators)


process.trackerCosmics_TopBot = cms.Sequence(process.trackerlocalrecoTop+process.tracksP5Top+process.trackerlocalrecoBottom+process.tracksP5Bottom)


process.ckftracks_plus_pixelless = cms.Sequence(process.ckftracks+process.ctfTracksPixelLess)


process.p = cms.Path(process.TriggerSelectionSequence+process.offlineBeamSpot+process.MeasurementTrackerEvent+process.TrackRefitter+process.RefitterHighPuritySequence+process.ApeEstimatorSequence)


process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0),
    collateHistograms = cms.untracked.bool(False),
    enableMultiThread = cms.untracked.bool(False),
    forceResetOnBeginLumi = cms.untracked.bool(False),
    LSbasedMode = cms.untracked.bool(False),
    verboseQT = cms.untracked.int32(0)
)


process.MessageLogger = cms.Service("MessageLogger",
    suppressInfo = cms.untracked.vstring(),
    debugs = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    suppressDebug = cms.untracked.vstring(),
    cout = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    cerr_stats = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        output = cms.untracked.string('cerr'),
        optionalPSet = cms.untracked.bool(True)
    ),
    warnings = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    default = cms.untracked.PSet(

    ),
    statistics = cms.untracked.vstring('cerr_stats'),
    cerr = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(10),
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('INFO'),
        FwkJob = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        FwkSummary = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(10000000)
        ),
        optionalPSet = cms.untracked.bool(True),
        SectorBuilder = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        HitSelector = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        CalculateAPE = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        ApeEstimator = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        AlignmentTrackSelector = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    FrameworkJobReport = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        optionalPSet = cms.untracked.bool(True),
        FwkJob = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(10000000)
        )
    ),
    suppressWarning = cms.untracked.vstring(),
    errors = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('warnings', 
        'errors', 
        'infos', 
        'debugs', 
        'cout', 
        'cerr'),
    debugModules = cms.untracked.vstring(),
    infos = cms.untracked.PSet(
        optionalPSet = cms.untracked.bool(True),
        Root_NoDictionary = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        placeholder = cms.untracked.bool(True)
    ),
    categories = cms.untracked.vstring('FwkJob', 
        'FwkReport', 
        'FwkSummary', 
        'Root_NoDictionary', 
        'SectorBuilder', 
        'ResidualErrorBinning', 
        'HitSelector', 
        'CalculateAPE', 
        'ApeEstimator', 
        'TrackRefitter', 
        'AlignmentTrackSelector'),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport')
)


process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    horeco = cms.PSet(
        initialSeed = cms.untracked.uint32(541321),
        engineName = cms.untracked.string('TRandom3')
    ),
    paramMuons = cms.PSet(
        initialSeed = cms.untracked.uint32(54525),
        engineName = cms.untracked.string('TRandom3')
    ),
    saveFileName = cms.untracked.string(''),
    hbhereco = cms.PSet(
        initialSeed = cms.untracked.uint32(541321),
        engineName = cms.untracked.string('TRandom3')
    ),
    simSiStripDigiSimLink = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    externalLHEProducer = cms.PSet(
        initialSeed = cms.untracked.uint32(234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    famosPileUp = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),
    simMuonDTDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    siTrackerGaussianSmearingRecHits = cms.PSet(
        initialSeed = cms.untracked.uint32(24680),
        engineName = cms.untracked.string('TRandom3')
    ),
    ecalPreshowerRecHit = cms.PSet(
        initialSeed = cms.untracked.uint32(6541321),
        engineName = cms.untracked.string('TRandom3')
    ),
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simMuonRPCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    hiSignal = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    mixSimCaloHits = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),
    mix = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    VtxSmeared = cms.PSet(
        initialSeed = cms.untracked.uint32(98765432),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    LHCTransport = cms.PSet(
        initialSeed = cms.untracked.uint32(87654321),
        engineName = cms.untracked.string('TRandom3')
    ),
    ecalRecHit = cms.PSet(
        initialSeed = cms.untracked.uint32(654321),
        engineName = cms.untracked.string('TRandom3')
    ),
    hfreco = cms.PSet(
        initialSeed = cms.untracked.uint32(541321),
        engineName = cms.untracked.string('TRandom3')
    ),
    mixRecoTracks = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),
    hiSignalG4SimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(11),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    famosSimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(13579),
        engineName = cms.untracked.string('TRandom3')
    ),
    MuonSimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(987346),
        engineName = cms.untracked.string('TRandom3')
    ),
    g4SimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(11),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    hiSignalLHCTransport = cms.PSet(
        initialSeed = cms.untracked.uint32(88776655),
        engineName = cms.untracked.string('TRandom3')
    ),
    mixGenPU = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),
    l1ParamMuons = cms.PSet(
        initialSeed = cms.untracked.uint32(6453209),
        engineName = cms.untracked.string('TRandom3')
    ),
    simBeamSpotFilter = cms.PSet(
        initialSeed = cms.untracked.uint32(87654321),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simMuonCSCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(11223344),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    mixData = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)


process.TFileService = cms.Service("TFileService",
    closeFileFast = cms.untracked.bool(True),
    fileName = cms.string('/afs/cern.ch/work/a/ajkumar/APE_newCPE_v1/CMSSW_7_2_0_pre6/src/ApeEstimator/ApeEstimator/hists/workingArea/data11.root')
)


process.AnalyticalPropagator = cms.ESProducer("AnalyticalPropagatorESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('AnalyticalPropagator'),
    PropagationDirection = cms.string('alongMomentum')
)


process.AnalyticalPropagatorParabolicMF = cms.ESProducer("AnalyticalPropagatorESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('AnalyticalPropagatorParabolicMf'),
    SimpleMagneticField = cms.string('ParabolicMf'),
    PropagationDirection = cms.string('alongMomentum')
)


process.AnyDirectionAnalyticalPropagator = cms.ESProducer("AnalyticalPropagatorESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('AnyDirectionAnalyticalPropagator'),
    PropagationDirection = cms.string('anyDirection')
)


process.BeamHaloMPropagatorAlong = cms.ESProducer("PropagatorWithMaterialESProducer",
    SimpleMagneticField = cms.string(''),
    PropagationDirection = cms.string('alongMomentum'),
    ComponentName = cms.string('BeamHaloMPropagatorAlong'),
    Mass = cms.double(0.105),
    ptMin = cms.double(-1.0),
    MaxDPhi = cms.double(10000),
    useRungeKutta = cms.bool(True)
)


process.BeamHaloMPropagatorOpposite = cms.ESProducer("PropagatorWithMaterialESProducer",
    SimpleMagneticField = cms.string(''),
    PropagationDirection = cms.string('oppositeToMomentum'),
    ComponentName = cms.string('BeamHaloMPropagatorOpposite'),
    Mass = cms.double(0.105),
    ptMin = cms.double(-1.0),
    MaxDPhi = cms.double(10000),
    useRungeKutta = cms.bool(True)
)


process.BeamHaloPropagatorAlong = cms.ESProducer("BeamHaloPropagatorESProducer",
    ComponentName = cms.string('BeamHaloPropagatorAlong'),
    CrossingTrackerPropagator = cms.string('BeamHaloSHPropagatorAlong'),
    PropagationDirection = cms.string('alongMomentum'),
    EndCapTrackerPropagator = cms.string('BeamHaloMPropagatorAlong')
)


process.BeamHaloPropagatorAny = cms.ESProducer("BeamHaloPropagatorESProducer",
    ComponentName = cms.string('BeamHaloPropagatorAny'),
    CrossingTrackerPropagator = cms.string('BeamHaloSHPropagatorAny'),
    PropagationDirection = cms.string('anyDirection'),
    EndCapTrackerPropagator = cms.string('BeamHaloMPropagatorAlong')
)


process.BeamHaloPropagatorOpposite = cms.ESProducer("BeamHaloPropagatorESProducer",
    ComponentName = cms.string('BeamHaloPropagatorOpposite'),
    CrossingTrackerPropagator = cms.string('BeamHaloSHPropagatorOpposite'),
    PropagationDirection = cms.string('oppositeToMomentum'),
    EndCapTrackerPropagator = cms.string('BeamHaloMPropagatorOpposite')
)


process.BeamHaloSHPropagatorAlong = cms.ESProducer("SteppingHelixPropagatorESProducer",
    endcapShiftInZNeg = cms.double(0.0),
    PropagationDirection = cms.string('alongMomentum'),
    useMatVolumes = cms.bool(True),
    useTuningForL2Speed = cms.bool(False),
    useIsYokeFlag = cms.bool(True),
    NoErrorPropagation = cms.bool(False),
    SetVBFPointer = cms.bool(False),
    AssumeNoMaterial = cms.bool(False),
    returnTangentPlane = cms.bool(True),
    useInTeslaFromMagField = cms.bool(False),
    VBFName = cms.string('VolumeBasedMagneticField'),
    useEndcapShiftsInZ = cms.bool(False),
    sendLogWarning = cms.bool(False),
    ComponentName = cms.string('BeamHaloSHPropagatorAlong'),
    debug = cms.bool(False),
    ApplyRadX0Correction = cms.bool(True),
    useMagVolumes = cms.bool(True),
    endcapShiftInZPos = cms.double(0.0)
)


process.BeamHaloSHPropagatorAny = cms.ESProducer("SteppingHelixPropagatorESProducer",
    endcapShiftInZNeg = cms.double(0.0),
    PropagationDirection = cms.string('anyDirection'),
    useMatVolumes = cms.bool(True),
    useTuningForL2Speed = cms.bool(False),
    useIsYokeFlag = cms.bool(True),
    NoErrorPropagation = cms.bool(False),
    SetVBFPointer = cms.bool(False),
    AssumeNoMaterial = cms.bool(False),
    returnTangentPlane = cms.bool(True),
    useInTeslaFromMagField = cms.bool(False),
    VBFName = cms.string('VolumeBasedMagneticField'),
    useEndcapShiftsInZ = cms.bool(False),
    sendLogWarning = cms.bool(False),
    ComponentName = cms.string('BeamHaloSHPropagatorAny'),
    debug = cms.bool(False),
    ApplyRadX0Correction = cms.bool(True),
    useMagVolumes = cms.bool(True),
    endcapShiftInZPos = cms.double(0.0)
)


process.BeamHaloSHPropagatorOpposite = cms.ESProducer("SteppingHelixPropagatorESProducer",
    endcapShiftInZNeg = cms.double(0.0),
    PropagationDirection = cms.string('oppositeToMomentum'),
    useMatVolumes = cms.bool(True),
    useTuningForL2Speed = cms.bool(False),
    useIsYokeFlag = cms.bool(True),
    NoErrorPropagation = cms.bool(False),
    SetVBFPointer = cms.bool(False),
    AssumeNoMaterial = cms.bool(False),
    returnTangentPlane = cms.bool(True),
    useInTeslaFromMagField = cms.bool(False),
    VBFName = cms.string('VolumeBasedMagneticField'),
    useEndcapShiftsInZ = cms.bool(False),
    sendLogWarning = cms.bool(False),
    ComponentName = cms.string('BeamHaloSHPropagatorOpposite'),
    debug = cms.bool(False),
    ApplyRadX0Correction = cms.bool(True),
    useMagVolumes = cms.bool(True),
    endcapShiftInZPos = cms.double(0.0)
)


process.CSCGeometryESModule = cms.ESProducer("CSCGeometryESModule",
    appendToDataLabel = cms.string(''),
    useDDD = cms.bool(True),
    debugV = cms.untracked.bool(False),
    useGangedStripsInME1a = cms.bool(True),
    alignmentsLabel = cms.string(''),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True),
    useCentreTIOffsets = cms.bool(False),
    applyAlignment = cms.bool(True)
)


process.CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL', 
        'ZDC', 
        'CASTOR', 
        'EcalBarrel', 
        'EcalEndcap', 
        'EcalPreshower', 
        'TOWER')
)


process.CaloTopologyBuilder = cms.ESProducer("CaloTopologyBuilder")


process.CaloTowerHardcodeGeometryEP = cms.ESProducer("CaloTowerHardcodeGeometryEP")


process.CastorDbProducer = cms.ESProducer("CastorDbProducer")


process.CastorHardcodeGeometryEP = cms.ESProducer("CastorHardcodeGeometryEP")


process.Chi2MeasurementEstimator = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    MaxChi2 = cms.double(30.0),
    nSigma = cms.double(3.0),
    ComponentName = cms.string('Chi2')
)


process.ClusterShapeHitFilterESProducer = cms.ESProducer("ClusterShapeHitFilterESProducer",
    ComponentName = cms.string('ClusterShapeHitFilter'),
    PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par')
)


process.DTGeometryESModule = cms.ESProducer("DTGeometryESModule",
    appendToDataLabel = cms.string(''),
    fromDDD = cms.bool(True),
    applyAlignment = cms.bool(True),
    alignmentsLabel = cms.string('')
)


process.DummyDetLayerGeometry = cms.ESProducer("DetLayerGeometryESProducer",
    ComponentName = cms.string('DummyDetLayerGeometry')
)


process.EcalBarrelGeometryEP = cms.ESProducer("EcalBarrelGeometryEP",
    applyAlignment = cms.bool(False)
)


process.EcalElectronicsMappingBuilder = cms.ESProducer("EcalElectronicsMappingBuilder")


process.EcalEndcapGeometryEP = cms.ESProducer("EcalEndcapGeometryEP",
    applyAlignment = cms.bool(False)
)


process.EcalLaserCorrectionService = cms.ESProducer("EcalLaserCorrectionService")


process.EcalPreshowerGeometryEP = cms.ESProducer("EcalPreshowerGeometryEP",
    applyAlignment = cms.bool(False)
)


process.EcalTrigTowerConstituentsMapBuilder = cms.ESProducer("EcalTrigTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/EcalMapping/data/EndCap_TTMap.txt')
)


process.FittingSmootherRKP5 = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(25.0),
    LogPixelProbabilityCut = cms.double(-14.0),
    Fitter = cms.string('RKFitter'),
    MinNumberOfHits = cms.int32(4),
    Smoother = cms.string('RKSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(False),
    ComponentName = cms.string('FittingSmootherRKP5'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.FlexibleKFFittingSmoother = cms.ESProducer("FlexibleKFFittingSmootherESProducer",
    ComponentName = cms.string('FlexibleKFFittingSmoother'),
    standardFitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    looperFitter = cms.string('LooperFittingSmoother')
)


process.GlobalDetLayerGeometry = cms.ESProducer("GlobalDetLayerGeometryESProducer",
    ComponentName = cms.string('GlobalDetLayerGeometry')
)


process.GlobalTrackingGeometryESProducer = cms.ESProducer("GlobalTrackingGeometryESProducer")


process.HcalHardcodeGeometryEP = cms.ESProducer("HcalHardcodeGeometryEP",
    HcalReLabel = cms.PSet(
        RelabelRules = cms.untracked.PSet(
            Eta16 = cms.untracked.vint32(1, 1, 2, 2, 2, 
                2, 2, 2, 2, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3),
            Eta17 = cms.untracked.vint32(1, 1, 2, 2, 3, 
                3, 3, 4, 4, 4, 
                4, 4, 5, 5, 5, 
                5, 5, 5, 5),
            Eta1 = cms.untracked.vint32(1, 2, 2, 2, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3),
            CorrectPhi = cms.untracked.bool(False)
        ),
        RelabelHits = cms.untracked.bool(False)
    )
)


process.KFFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    LogPixelProbabilityCut = cms.double(-16.0),
    Fitter = cms.string('KFFitter'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('KFSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('KFFittingSmoother'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.KFFittingSmootherBeamHalo = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    LogPixelProbabilityCut = cms.double(-16.0),
    Fitter = cms.string('KFFitterBH'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('KFSmootherBH'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('KFFittingSmootherBH'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.KFFittingSmootherWithOutliersRejectionAndRK = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(20.0),
    LogPixelProbabilityCut = cms.double(-14.0),
    Fitter = cms.string('RKFitter'),
    MinNumberOfHits = cms.int32(3),
    Smoother = cms.string('RKSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.KFSwitching1DUpdatorESProducer = cms.ESProducer("KFSwitching1DUpdatorESProducer",
    ComponentName = cms.string('KFSwitching1DUpdator'),
    doEndCap = cms.bool(False)
)


process.KFTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    ComponentName = cms.string('KFFitter'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('PropagatorWithMaterial'),
    minHits = cms.int32(3)
)


process.KFTrajectoryFitterBeamHalo = cms.ESProducer("KFTrajectoryFitterESProducer",
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    ComponentName = cms.string('KFFitterBH'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('BeamHaloPropagatorAlong'),
    minHits = cms.int32(3)
)


process.KFTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmoother'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('PropagatorWithMaterial'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)


process.KFTrajectorySmootherBeamHalo = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmootherBH'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('BeamHaloPropagatorAlong'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)


process.KFUpdatorESProducer = cms.ESProducer("KFUpdatorESProducer",
    ComponentName = cms.string('KFUpdator')
)


process.LooperFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(20.0),
    LogPixelProbabilityCut = cms.double(-14.0),
    Fitter = cms.string('LooperFitter'),
    MinNumberOfHits = cms.int32(3),
    Smoother = cms.string('LooperSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('LooperFittingSmoother'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.LooperTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    ComponentName = cms.string('LooperFitter'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers'),
    minHits = cms.int32(3)
)


process.LooperTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(10.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('LooperSmoother'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)


process.MRHChi2MeasurementEstimator = cms.ESProducer("MRHChi2MeasurementEstimatorESProducer",
    MaxChi2 = cms.double(30.0),
    nSigma = cms.double(3.0),
    ComponentName = cms.string('MRHChi2')
)


process.MRHFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    LogPixelProbabilityCut = cms.double(-16.0),
    Fitter = cms.string('MRHFitter'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('MRHSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('MRHFittingSmoother'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.MRHTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    ComponentName = cms.string('MRHFitter'),
    Estimator = cms.string('MRHChi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    minHits = cms.int32(3)
)


process.MRHTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('MRHSmoother'),
    Estimator = cms.string('MRHChi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)


process.MaterialPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    SimpleMagneticField = cms.string(''),
    PropagationDirection = cms.string('alongMomentum'),
    ComponentName = cms.string('PropagatorWithMaterial'),
    Mass = cms.double(0.105),
    ptMin = cms.double(-1.0),
    MaxDPhi = cms.double(1.6),
    useRungeKutta = cms.bool(False)
)


process.MaterialPropagatorParabolicMF = cms.ESProducer("PropagatorWithMaterialESProducer",
    SimpleMagneticField = cms.string('ParabolicMf'),
    PropagationDirection = cms.string('alongMomentum'),
    ComponentName = cms.string('PropagatorWithMaterialParabolicMf'),
    Mass = cms.double(0.105),
    ptMin = cms.double(-1.0),
    MaxDPhi = cms.double(1.6),
    useRungeKutta = cms.bool(False)
)


process.MeasurementTracker = cms.ESProducer("MeasurementTrackerESProducer",
    UseStripStripQualityDB = cms.bool(True),
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    UsePixelROCQualityDB = cms.bool(True),
    DebugPixelROCQualityDB = cms.untracked.bool(False),
    UseStripAPVFiberQualityDB = cms.bool(True),
    badStripCuts = cms.PSet(
        TOB = cms.PSet(
            maxConsecutiveBad = cms.uint32(2),
            maxBad = cms.uint32(4)
        ),
        TID = cms.PSet(
            maxConsecutiveBad = cms.uint32(2),
            maxBad = cms.uint32(4)
        ),
        TEC = cms.PSet(
            maxConsecutiveBad = cms.uint32(2),
            maxBad = cms.uint32(4)
        ),
        TIB = cms.PSet(
            maxConsecutiveBad = cms.uint32(2),
            maxBad = cms.uint32(4)
        )
    ),
    DebugStripModuleQualityDB = cms.untracked.bool(False),
    ComponentName = cms.string(''),
    DebugPixelModuleQualityDB = cms.untracked.bool(False),
    UsePixelModuleQualityDB = cms.bool(True),
    DebugStripAPVFiberQualityDB = cms.untracked.bool(False),
    HitMatcher = cms.string('StandardMatcher'),
    DebugStripStripQualityDB = cms.untracked.bool(False),
    UseStripModuleQualityDB = cms.bool(True),
    SiStripQualityLabel = cms.string(''),
    MaskBadAPVFibers = cms.bool(True),
    PixelCPE = cms.string('PixelCPEGeneric')
)


process.MeasurementTrackerBottom = cms.ESProducer("MeasurementTrackerESProducer",
    UseStripStripQualityDB = cms.bool(True),
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    UsePixelROCQualityDB = cms.bool(True),
    DebugPixelROCQualityDB = cms.untracked.bool(False),
    UseStripAPVFiberQualityDB = cms.bool(True),
    badStripCuts = cms.PSet(
        TOB = cms.PSet(
            maxConsecutiveBad = cms.uint32(2),
            maxBad = cms.uint32(4)
        ),
        TID = cms.PSet(
            maxConsecutiveBad = cms.uint32(2),
            maxBad = cms.uint32(4)
        ),
        TEC = cms.PSet(
            maxConsecutiveBad = cms.uint32(2),
            maxBad = cms.uint32(4)
        ),
        TIB = cms.PSet(
            maxConsecutiveBad = cms.uint32(2),
            maxBad = cms.uint32(4)
        )
    ),
    DebugStripModuleQualityDB = cms.untracked.bool(False),
    ComponentName = cms.string('MeasurementTrackerBottom'),
    stripClusterProducer = cms.string('siStripClustersBottom'),
    DebugPixelModuleQualityDB = cms.untracked.bool(False),
    UsePixelModuleQualityDB = cms.bool(True),
    DebugStripAPVFiberQualityDB = cms.untracked.bool(False),
    HitMatcher = cms.string('StandardMatcher'),
    DebugStripStripQualityDB = cms.untracked.bool(False),
    PixelCPE = cms.string('PixelCPEGeneric'),
    pixelClusterProducer = cms.string('siPixelClustersBottom'),
    SiStripQualityLabel = cms.string(''),
    UseStripModuleQualityDB = cms.bool(True),
    MaskBadAPVFibers = cms.bool(True)
)


process.MeasurementTrackerTop = cms.ESProducer("MeasurementTrackerESProducer",
    UseStripStripQualityDB = cms.bool(True),
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    UsePixelROCQualityDB = cms.bool(True),
    DebugPixelROCQualityDB = cms.untracked.bool(False),
    UseStripAPVFiberQualityDB = cms.bool(True),
    badStripCuts = cms.PSet(
        TOB = cms.PSet(
            maxConsecutiveBad = cms.uint32(2),
            maxBad = cms.uint32(4)
        ),
        TID = cms.PSet(
            maxConsecutiveBad = cms.uint32(2),
            maxBad = cms.uint32(4)
        ),
        TEC = cms.PSet(
            maxConsecutiveBad = cms.uint32(2),
            maxBad = cms.uint32(4)
        ),
        TIB = cms.PSet(
            maxConsecutiveBad = cms.uint32(2),
            maxBad = cms.uint32(4)
        )
    ),
    DebugStripModuleQualityDB = cms.untracked.bool(False),
    ComponentName = cms.string('MeasurementTrackerTop'),
    stripClusterProducer = cms.string('siStripClustersTop'),
    DebugPixelModuleQualityDB = cms.untracked.bool(False),
    UsePixelModuleQualityDB = cms.bool(True),
    DebugStripAPVFiberQualityDB = cms.untracked.bool(False),
    HitMatcher = cms.string('StandardMatcher'),
    DebugStripStripQualityDB = cms.untracked.bool(False),
    PixelCPE = cms.string('PixelCPEGeneric'),
    pixelClusterProducer = cms.string('siPixelClustersTop'),
    SiStripQualityLabel = cms.string(''),
    UseStripModuleQualityDB = cms.bool(True),
    MaskBadAPVFibers = cms.bool(True)
)


process.MuonDetLayerGeometryESProducer = cms.ESProducer("MuonDetLayerGeometryESProducer")


process.MuonNumberingInitialization = cms.ESProducer("MuonNumberingInitialization")


process.OppositeAnalyticalPropagator = cms.ESProducer("AnalyticalPropagatorESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('AnalyticalPropagatorOpposite'),
    PropagationDirection = cms.string('oppositeToMomentum')
)


process.OppositeAnalyticalPropagatorParabolicMF = cms.ESProducer("AnalyticalPropagatorESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('AnalyticalPropagatorParabolicMfOpposite'),
    SimpleMagneticField = cms.string('ParabolicMf'),
    PropagationDirection = cms.string('oppositeToMomentum')
)


process.OppositeMaterialPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    SimpleMagneticField = cms.string(''),
    PropagationDirection = cms.string('oppositeToMomentum'),
    ComponentName = cms.string('PropagatorWithMaterialOpposite'),
    Mass = cms.double(0.105),
    ptMin = cms.double(-1.0),
    MaxDPhi = cms.double(1.6),
    useRungeKutta = cms.bool(False)
)


process.OppositeMaterialPropagatorParabolicMF = cms.ESProducer("PropagatorWithMaterialESProducer",
    SimpleMagneticField = cms.string('ParabolicMf'),
    PropagationDirection = cms.string('oppositeToMomentum'),
    ComponentName = cms.string('PropagatorWithMaterialParabolicMfOpposite'),
    Mass = cms.double(0.105),
    ptMin = cms.double(-1.0),
    MaxDPhi = cms.double(1.6),
    useRungeKutta = cms.bool(False)
)


process.ParabolicParametrizedMagneticFieldProducer = cms.ESProducer("ParametrizedMagneticFieldProducer",
    version = cms.string('Parabolic'),
    parameters = cms.PSet(

    ),
    label = cms.untracked.string('ParabolicMf')
)


process.ParametrizedMagneticFieldProducer = cms.ESProducer("ParametrizedMagneticFieldProducer",
    version = cms.string('OAE_1103l_071212'),
    parameters = cms.PSet(
        BValue = cms.string('3_8T')
    ),
    label = cms.untracked.string('parametrizedField')
)


process.PixelCPEGenericESProducer = cms.ESProducer("PixelCPEGenericESProducer",
    useLAAlignmentOffsets = cms.bool(False),
    DoCosmics = cms.bool(False),
    eff_charge_cut_highX = cms.double(1.0),
    eff_charge_cut_highY = cms.double(1.0),
    inflate_all_errors_no_trk_angle = cms.bool(False),
    eff_charge_cut_lowY = cms.double(0.0),
    eff_charge_cut_lowX = cms.double(0.0),
    UseErrorsFromTemplates = cms.bool(True),
    TruncatePixelCharge = cms.bool(True),
    size_cutY = cms.double(3.0),
    size_cutX = cms.double(3.0),
    useLAWidthFromDB = cms.bool(False),
    inflate_errors = cms.bool(False),
    Alpha2Order = cms.bool(True),
    ClusterProbComputationFlag = cms.int32(0),
    PixelErrorParametrization = cms.string('NOTcmsim'),
    EdgeClusterErrorX = cms.double(50.0),
    EdgeClusterErrorY = cms.double(85.0),
    LoadTemplatesFromDB = cms.bool(True),
    ComponentName = cms.string('PixelCPEGeneric'),
    MagneticFieldRecord = cms.ESInputTag(""),
    IrradiationBiasCorrection = cms.bool(False)
)


process.PropagatorWithMaterialForLoopers = cms.ESProducer("PropagatorWithMaterialESProducer",
    useOldAnalPropLogic = cms.bool(False),
    SimpleMagneticField = cms.string(''),
    PropagationDirection = cms.string('alongMomentum'),
    ComponentName = cms.string('PropagatorWithMaterialForLoopers'),
    Mass = cms.double(0.1396),
    ptMin = cms.double(-1),
    MaxDPhi = cms.double(4.0),
    useRungeKutta = cms.bool(False)
)


process.PropagatorWithMaterialForLoopersOpposite = cms.ESProducer("PropagatorWithMaterialESProducer",
    useOldAnalPropLogic = cms.bool(False),
    SimpleMagneticField = cms.string(''),
    PropagationDirection = cms.string('oppositeToMomentum'),
    ComponentName = cms.string('PropagatorWithMaterialForLoopersOpposite'),
    Mass = cms.double(0.1396),
    ptMin = cms.double(-1),
    MaxDPhi = cms.double(4.0),
    useRungeKutta = cms.bool(False)
)


process.RK1DFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    LogPixelProbabilityCut = cms.double(-16.0),
    Fitter = cms.string('RK1DFitter'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('RK1DSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('RK1DFittingSmoother'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.RK1DTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    ComponentName = cms.string('RK1DFitter'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFSwitching1DUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    minHits = cms.int32(3)
)


process.RK1DTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('RK1DSmoother'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFSwitching1DUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)


process.RKFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    LogPixelProbabilityCut = cms.double(-16.0),
    Fitter = cms.string('RKFitter'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('RKSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('RKFittingSmoother'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.RKOutliers1DFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(20.0),
    LogPixelProbabilityCut = cms.double(-16.0),
    Fitter = cms.string('RK1DFitter'),
    MinNumberOfHits = cms.int32(3),
    Smoother = cms.string('RK1DSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('RKOutliers1DFittingSmoother'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.RKTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    ComponentName = cms.string('RKFitter'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    minHits = cms.int32(3)
)


process.RKTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('RKSmoother'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)


process.RPCGeometryESModule = cms.ESProducer("RPCGeometryESModule",
    useDDD = cms.untracked.bool(True),
    compatibiltyWith11 = cms.untracked.bool(True)
)


process.RungeKuttaTrackerPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    SimpleMagneticField = cms.string(''),
    PropagationDirection = cms.string('alongMomentum'),
    ComponentName = cms.string('RungeKuttaTrackerPropagator'),
    Mass = cms.double(0.105),
    ptMin = cms.double(-1.0),
    MaxDPhi = cms.double(1.6),
    useRungeKutta = cms.bool(True)
)


process.SiStripRecHitMatcherESProducer = cms.ESProducer("SiStripRecHitMatcherESProducer",
    PreFilter = cms.bool(False),
    ComponentName = cms.string('StandardMatcher'),
    NSigmaInside = cms.double(3.0)
)


process.SteppingHelixPropagatorAlong = cms.ESProducer("SteppingHelixPropagatorESProducer",
    endcapShiftInZNeg = cms.double(0.0),
    PropagationDirection = cms.string('alongMomentum'),
    useMatVolumes = cms.bool(True),
    useTuningForL2Speed = cms.bool(False),
    useIsYokeFlag = cms.bool(True),
    NoErrorPropagation = cms.bool(False),
    SetVBFPointer = cms.bool(False),
    AssumeNoMaterial = cms.bool(False),
    returnTangentPlane = cms.bool(True),
    useInTeslaFromMagField = cms.bool(False),
    VBFName = cms.string('VolumeBasedMagneticField'),
    useEndcapShiftsInZ = cms.bool(False),
    sendLogWarning = cms.bool(False),
    ComponentName = cms.string('SteppingHelixPropagatorAlong'),
    debug = cms.bool(False),
    ApplyRadX0Correction = cms.bool(True),
    useMagVolumes = cms.bool(True),
    endcapShiftInZPos = cms.double(0.0)
)


process.SteppingHelixPropagatorAny = cms.ESProducer("SteppingHelixPropagatorESProducer",
    endcapShiftInZNeg = cms.double(0.0),
    PropagationDirection = cms.string('anyDirection'),
    useMatVolumes = cms.bool(True),
    useTuningForL2Speed = cms.bool(False),
    useIsYokeFlag = cms.bool(True),
    NoErrorPropagation = cms.bool(False),
    SetVBFPointer = cms.bool(False),
    AssumeNoMaterial = cms.bool(False),
    returnTangentPlane = cms.bool(True),
    useInTeslaFromMagField = cms.bool(False),
    VBFName = cms.string('VolumeBasedMagneticField'),
    useEndcapShiftsInZ = cms.bool(False),
    sendLogWarning = cms.bool(False),
    ComponentName = cms.string('SteppingHelixPropagatorAny'),
    debug = cms.bool(False),
    ApplyRadX0Correction = cms.bool(True),
    useMagVolumes = cms.bool(True),
    endcapShiftInZPos = cms.double(0.0)
)


process.SteppingHelixPropagatorOpposite = cms.ESProducer("SteppingHelixPropagatorESProducer",
    endcapShiftInZNeg = cms.double(0.0),
    PropagationDirection = cms.string('oppositeToMomentum'),
    useMatVolumes = cms.bool(True),
    useTuningForL2Speed = cms.bool(False),
    useIsYokeFlag = cms.bool(True),
    NoErrorPropagation = cms.bool(False),
    SetVBFPointer = cms.bool(False),
    AssumeNoMaterial = cms.bool(False),
    returnTangentPlane = cms.bool(True),
    useInTeslaFromMagField = cms.bool(False),
    VBFName = cms.string('VolumeBasedMagneticField'),
    useEndcapShiftsInZ = cms.bool(False),
    sendLogWarning = cms.bool(False),
    ComponentName = cms.string('SteppingHelixPropagatorOpposite'),
    debug = cms.bool(False),
    ApplyRadX0Correction = cms.bool(True),
    useMagVolumes = cms.bool(True),
    endcapShiftInZPos = cms.double(0.0)
)


process.StripCPEESProducer = cms.ESProducer("StripCPEESProducer",
    ComponentType = cms.string('SimpleStripCPE'),
    ComponentName = cms.string('SimpleStripCPE')
)


process.StripCPEfromTrackAngleESProducer = cms.ESProducer("StripCPEESProducer",
    mTOB_P0 = cms.double(-1.026),
    mLC_P2 = cms.double(0.3),
    mLC_P1 = cms.double(0.618),
    mLC_P0 = cms.double(-0.326),
    useLegacyError = cms.bool(False),
    ComponentName = cms.string('StripCPEfromTrackAngle'),
    mTEC_P1 = cms.double(0.471),
    mTEC_P0 = cms.double(-1.885),
    ComponentType = cms.string('StripCPEfromTrackAngle'),
    mTOB_P1 = cms.double(0.253),
    mTIB_P0 = cms.double(-0.742),
    mTIB_P1 = cms.double(0.202),
    mTID_P0 = cms.double(-1.427),
    mTID_P1 = cms.double(0.433)
)


process.StripCPEgeometricESProducer = cms.ESProducer("StripCPEESProducer",
    TanDiffusionAngle = cms.double(0.01),
    UncertaintyScaling = cms.double(1.42),
    ThicknessRelativeUncertainty = cms.double(0.02),
    MaybeNoiseThreshold = cms.double(3.5),
    ComponentName = cms.string('StripCPEgeometric'),
    MinimumUncertainty = cms.double(0.01),
    ComponentType = cms.string('StripCPEgeometric'),
    NoiseThreshold = cms.double(2.3)
)


process.TTRHBuilderAngleAndTemplate = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    PixelCPE = cms.string('PixelCPETemplateReco'),
    ComponentName = cms.string('WithAngleAndTemplate')
)


process.TTRHBuilderGeometricAndTemplate = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    PixelCPE = cms.string('PixelCPEGeneric'),
    ComponentName = cms.string('WithGeometricAndTemplate')
)


process.TrackerRecoGeometryESProducer = cms.ESProducer("TrackerRecoGeometryESProducer")


process.VolumeBasedMagneticFieldESProducer = cms.ESProducer("VolumeBasedMagneticFieldESProducer",
    scalingVolumes = cms.vint32(14100, 14200, 17600, 17800, 17900, 
        18100, 18300, 18400, 18600, 23100, 
        23300, 23400, 23600, 23800, 23900, 
        24100, 28600, 28800, 28900, 29100, 
        29300, 29400, 29600, 28609, 28809, 
        28909, 29109, 29309, 29409, 29609, 
        28610, 28810, 28910, 29110, 29310, 
        29410, 29610, 28611, 28811, 28911, 
        29111, 29311, 29411, 29611),
    scalingFactors = cms.vdouble(1, 1, 0.994, 1.004, 1.004, 
        1.005, 1.004, 1.004, 0.994, 0.965, 
        0.958, 0.958, 0.953, 0.958, 0.958, 
        0.965, 0.918, 0.924, 0.924, 0.906, 
        0.924, 0.924, 0.918, 0.991, 0.998, 
        0.998, 0.978, 0.998, 0.998, 0.991, 
        0.991, 0.998, 0.998, 0.978, 0.998, 
        0.998, 0.991, 0.991, 0.998, 0.998, 
        0.978, 0.998, 0.998, 0.991),
    useParametrizedTrackerField = cms.bool(True),
    label = cms.untracked.string(''),
    version = cms.string('grid_1103l_090322_3_8t'),
    debugBuilder = cms.untracked.bool(False),
    paramLabel = cms.string('parametrizedField'),
    geometryVersion = cms.int32(90322),
    gridFiles = cms.VPSet(cms.PSet(
        path = cms.string('grid.[v].bin'),
        master = cms.int32(1),
        volumes = cms.string('1-312'),
        sectors = cms.string('0')
    ), 
        cms.PSet(
            path = cms.string('S3/grid.[v].bin'),
            master = cms.int32(3),
            volumes = cms.string('176-186,231-241,286-296'),
            sectors = cms.string('3')
        ), 
        cms.PSet(
            path = cms.string('S4/grid.[v].bin'),
            master = cms.int32(4),
            volumes = cms.string('176-186,231-241,286-296'),
            sectors = cms.string('4')
        ), 
        cms.PSet(
            path = cms.string('S9/grid.[v].bin'),
            master = cms.int32(9),
            volumes = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors = cms.string('9')
        ), 
        cms.PSet(
            path = cms.string('S10/grid.[v].bin'),
            master = cms.int32(10),
            volumes = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors = cms.string('10')
        ), 
        cms.PSet(
            path = cms.string('S11/grid.[v].bin'),
            master = cms.int32(11),
            volumes = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors = cms.string('11')
        )),
    cacheLastVolume = cms.untracked.bool(True)
)


process.ZdcHardcodeGeometryEP = cms.ESProducer("ZdcHardcodeGeometryEP")


process.beamHaloNavigationSchoolESProducer = cms.ESProducer("NavigationSchoolESProducer",
    ComponentName = cms.string('BeamHaloNavigationSchool'),
    SimpleMagneticField = cms.string('')
)


process.conv2StepFitterSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(30),
    LogPixelProbabilityCut = cms.double(-14.0),
    Fitter = cms.string('RKFitter'),
    MinNumberOfHits = cms.int32(3),
    Smoother = cms.string('conv2StepRKSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('conv2StepFitterSmoother'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.conv2StepRKTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(10.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('conv2StepRKSmoother'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)


process.convStepFitterSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(30),
    LogPixelProbabilityCut = cms.double(-14.0),
    Fitter = cms.string('RKFitter'),
    MinNumberOfHits = cms.int32(3),
    Smoother = cms.string('convStepRKSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('convStepFitterSmoother'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.convStepRKTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(10.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('convStepRKSmoother'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)


process.cosmicsNavigationSchoolESProducer = cms.ESProducer("SkippingLayerCosmicNavigationSchoolESProducer",
    noPXB = cms.bool(False),
    noTID = cms.bool(False),
    noPXF = cms.bool(False),
    noTIB = cms.bool(False),
    ComponentName = cms.string('CosmicNavigationSchool'),
    allSelf = cms.bool(True),
    noTEC = cms.bool(False),
    noTOB = cms.bool(False),
    selfSearch = cms.bool(True)
)


process.detachedTripletStepChi2Est = cms.ESProducer("Chi2ChargeMeasurementEstimatorESProducer",
    minGoodStripCharge = cms.double(2069),
    MaxChi2 = cms.double(9.0),
    nSigma = cms.double(3.0),
    ComponentName = cms.string('detachedTripletStepChi2Est')
)


process.detachedTripletStepTrajectoryCleanerBySharedHits = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('detachedTripletStepTrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.13),
    ValidHitBonus = cms.double(5.0),
    ComponentType = cms.string('TrajectoryCleanerBySharedHits'),
    MissingHitPenalty = cms.double(20.0),
    allowSharedFirstHit = cms.bool(True)
)


process.fakeForIdealAlignment = cms.ESProducer("FakeAlignmentProducer",
    appendToDataLabel = cms.string('fakeForIdeal')
)


process.hcalTopologyIdeal = cms.ESProducer("HcalTopologyIdealEP",
    Exclude = cms.untracked.string(''),
    appendToDataLabel = cms.string(''),
    hcalTopologyConstants = cms.PSet(
        maxDepthHE = cms.int32(3),
        maxDepthHB = cms.int32(2),
        mode = cms.string('HcalTopologyMode::LHC')
    )
)


process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
    file = cms.untracked.string(''),
    dump = cms.untracked.vstring('')
)


process.hitCollectorForOutInMuonSeeds = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    MaxChi2 = cms.double(100.0),
    nSigma = cms.double(4.0),
    ComponentName = cms.string('hitCollectorForOutInMuonSeeds')
)


process.idealForDigiCSCGeometry = cms.ESProducer("CSCGeometryESModule",
    appendToDataLabel = cms.string('idealForDigi'),
    useDDD = cms.bool(True),
    debugV = cms.untracked.bool(False),
    useGangedStripsInME1a = cms.bool(True),
    alignmentsLabel = cms.string('fakeForIdeal'),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True),
    useCentreTIOffsets = cms.bool(False),
    applyAlignment = cms.bool(False)
)


process.idealForDigiDTGeometry = cms.ESProducer("DTGeometryESModule",
    appendToDataLabel = cms.string('idealForDigi'),
    fromDDD = cms.bool(True),
    applyAlignment = cms.bool(False),
    alignmentsLabel = cms.string('fakeForIdeal')
)


process.idealForDigiTrackerGeometry = cms.ESProducer("TrackerDigiGeometryESModule",
    appendToDataLabel = cms.string('idealForDigi'),
    fromDDD = cms.bool(True),
    trackerGeometryConstants = cms.PSet(
        ROCS_X = cms.int32(0),
        ROCS_Y = cms.int32(0),
        upgradeGeometry = cms.bool(False),
        BIG_PIX_PER_ROC_Y = cms.int32(2),
        BIG_PIX_PER_ROC_X = cms.int32(1),
        ROWS_PER_ROC = cms.int32(80),
        COLS_PER_ROC = cms.int32(52)
    ),
    applyAlignment = cms.bool(False),
    alignmentsLabel = cms.string('fakeForIdeal')
)


process.initialStepChi2Est = cms.ESProducer("Chi2ChargeMeasurementEstimatorESProducer",
    minGoodStripCharge = cms.double(1724),
    MaxChi2 = cms.double(30.0),
    pTChargeCutThreshold = cms.double(15.0),
    nSigma = cms.double(3.0),
    ComponentName = cms.string('initialStepChi2Est')
)


process.jetCoreRegionalStepChi2Est = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    MaxChi2 = cms.double(30.0),
    nSigma = cms.double(3.0),
    ComponentName = cms.string('jetCoreRegionalStepChi2Est')
)


process.lowPtTripletStepChi2Est = cms.ESProducer("Chi2ChargeMeasurementEstimatorESProducer",
    minGoodStripCharge = cms.double(2069),
    MaxChi2 = cms.double(9.0),
    nSigma = cms.double(3.0),
    ComponentName = cms.string('lowPtTripletStepChi2Est')
)


process.lowPtTripletStepTrajectoryCleanerBySharedHits = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('lowPtTripletStepTrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.16),
    ValidHitBonus = cms.double(5.0),
    ComponentType = cms.string('TrajectoryCleanerBySharedHits'),
    MissingHitPenalty = cms.double(20.0),
    allowSharedFirstHit = cms.bool(True)
)


process.mixedTripletStepChi2Est = cms.ESProducer("Chi2ChargeMeasurementEstimatorESProducer",
    minGoodStripCharge = cms.double(2069),
    MaxChi2 = cms.double(16.0),
    nSigma = cms.double(3.0),
    ComponentName = cms.string('mixedTripletStepChi2Est')
)


process.mixedTripletStepClusterShapeHitFilter = cms.ESProducer("ClusterShapeHitFilterESProducer",
    ComponentName = cms.string('mixedTripletStepClusterShapeHitFilter'),
    PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par'),
    minGoodStripCharge = cms.double(2069)
)


process.mixedTripletStepPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    SimpleMagneticField = cms.string(''),
    PropagationDirection = cms.string('alongMomentum'),
    ComponentName = cms.string('mixedTripletStepPropagator'),
    Mass = cms.double(0.105),
    ptMin = cms.double(0.1),
    MaxDPhi = cms.double(1.6),
    useRungeKutta = cms.bool(False)
)


process.mixedTripletStepPropagatorOpposite = cms.ESProducer("PropagatorWithMaterialESProducer",
    SimpleMagneticField = cms.string(''),
    PropagationDirection = cms.string('oppositeToMomentum'),
    ComponentName = cms.string('mixedTripletStepPropagatorOpposite'),
    Mass = cms.double(0.105),
    ptMin = cms.double(0.1),
    MaxDPhi = cms.double(1.6),
    useRungeKutta = cms.bool(False)
)


process.mixedTripletStepTrajectoryCleanerBySharedHits = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('mixedTripletStepTrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.11),
    ValidHitBonus = cms.double(5.0),
    ComponentType = cms.string('TrajectoryCleanerBySharedHits'),
    MissingHitPenalty = cms.double(20.0),
    allowSharedFirstHit = cms.bool(True)
)


process.muonSeededFittingSmootherWithOutliersRejectionAndRK = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(50.0),
    LogPixelProbabilityCut = cms.double(-14.0),
    Fitter = cms.string('RKFitter'),
    MinNumberOfHits = cms.int32(3),
    Smoother = cms.string('RKSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(False),
    ComponentName = cms.string('muonSeededFittingSmootherWithOutliersRejectionAndRK'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.muonSeededMeasurementEstimatorForInOut = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    MaxChi2 = cms.double(400.0),
    nSigma = cms.double(4.0),
    ComponentName = cms.string('muonSeededMeasurementEstimatorForInOut')
)


process.muonSeededMeasurementEstimatorForOutIn = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    MaxChi2 = cms.double(30.0),
    nSigma = cms.double(3.0),
    ComponentName = cms.string('muonSeededMeasurementEstimatorForOutIn')
)


process.muonSeededTrajectoryCleanerBySharedHits = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('muonSeededTrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.1),
    ValidHitBonus = cms.double(1000.0),
    ComponentType = cms.string('TrajectoryCleanerBySharedHits'),
    MissingHitPenalty = cms.double(1.0),
    allowSharedFirstHit = cms.bool(True)
)


process.myTTRHBuilderWithoutAngle4MixedPairs = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    Matcher = cms.string('StandardMatcher'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    PixelCPE = cms.string('PixelCPEGeneric'),
    ComponentName = cms.string('TTRHBuilderWithoutAngle4MixedPairs')
)


process.myTTRHBuilderWithoutAngle4MixedTriplets = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    Matcher = cms.string('StandardMatcher'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    PixelCPE = cms.string('PixelCPEGeneric'),
    ComponentName = cms.string('TTRHBuilderWithoutAngle4MixedTriplets')
)


process.myTTRHBuilderWithoutAngle4PixelPairs = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    Matcher = cms.string('StandardMatcher'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    PixelCPE = cms.string('PixelCPEGeneric'),
    ComponentName = cms.string('TTRHBuilderWithoutAngle4PixelPairs')
)


process.myTTRHBuilderWithoutAngle4PixelTriplets = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    Matcher = cms.string('StandardMatcher'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    PixelCPE = cms.string('PixelCPEGeneric'),
    ComponentName = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
)


process.navigationSchoolESProducer = cms.ESProducer("NavigationSchoolESProducer",
    ComponentName = cms.string('SimpleNavigationSchool'),
    SimpleMagneticField = cms.string('')
)


process.pixelLessStepChi2Est = cms.ESProducer("Chi2ChargeMeasurementEstimatorESProducer",
    minGoodStripCharge = cms.double(2069),
    MaxChi2 = cms.double(9.0),
    nSigma = cms.double(3.0),
    ComponentName = cms.string('pixelLessStepChi2Est')
)


process.pixelLessStepClusterShapeHitFilter = cms.ESProducer("ClusterShapeHitFilterESProducer",
    ComponentName = cms.string('pixelLessStepClusterShapeHitFilter'),
    PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par'),
    minGoodStripCharge = cms.double(2069)
)


process.pixelLessStepTrajectoryCleanerBySharedHits = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('pixelLessStepTrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.11),
    ValidHitBonus = cms.double(5.0),
    ComponentType = cms.string('TrajectoryCleanerBySharedHits'),
    MissingHitPenalty = cms.double(20.0),
    allowSharedFirstHit = cms.bool(True)
)


process.pixelPairStepChi2Est = cms.ESProducer("Chi2ChargeMeasurementEstimatorESProducer",
    minGoodStripCharge = cms.double(2069),
    MaxChi2 = cms.double(9.0),
    pTChargeCutThreshold = cms.double(15.0),
    nSigma = cms.double(3.0),
    ComponentName = cms.string('pixelPairStepChi2Est')
)


process.siPixelQualityESProducer = cms.ESProducer("SiPixelQualityESProducer",
    ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelQualityFromDbRcd'),
        tag = cms.string('')
    ), 
        cms.PSet(
            record = cms.string('SiPixelDetVOffRcd'),
            tag = cms.string('')
        ))
)


process.siPixelTemplateDBObjectESProducer = cms.ESProducer("SiPixelTemplateDBObjectESProducer")


process.siStripBackPlaneCorrectionDepESProducer = cms.ESProducer("SiStripBackPlaneCorrectionDepESProducer",
    LatencyRecord = cms.PSet(
        record = cms.string('SiStripLatencyRcd'),
        label = cms.untracked.string('')
    ),
    BackPlaneCorrectionDeconvMode = cms.PSet(
        record = cms.string('SiStripBackPlaneCorrectionRcd'),
        label = cms.untracked.string('deconvolution')
    ),
    BackPlaneCorrectionPeakMode = cms.PSet(
        record = cms.string('SiStripBackPlaneCorrectionRcd'),
        label = cms.untracked.string('peak')
    )
)


process.siStripGainESProducer = cms.ESProducer("SiStripGainESProducer",
    printDebug = cms.untracked.bool(False),
    appendToDataLabel = cms.string(''),
    APVGain = cms.VPSet(cms.PSet(
        Record = cms.string('SiStripApvGainRcd'),
        NormalizationFactor = cms.untracked.double(1.0),
        Label = cms.untracked.string('')
    ), 
        cms.PSet(
            Record = cms.string('SiStripApvGain2Rcd'),
            NormalizationFactor = cms.untracked.double(1.0),
            Label = cms.untracked.string('')
        )),
    AutomaticNormalization = cms.bool(False)
)


process.siStripLorentzAngleDepESProducer = cms.ESProducer("SiStripLorentzAngleDepESProducer",
    LatencyRecord = cms.PSet(
        record = cms.string('SiStripLatencyRcd'),
        label = cms.untracked.string('')
    ),
    LorentzAngleDeconvMode = cms.PSet(
        record = cms.string('SiStripLorentzAngleRcd'),
        label = cms.untracked.string('deconvolution')
    ),
    LorentzAnglePeakMode = cms.PSet(
        record = cms.string('SiStripLorentzAngleRcd'),
        label = cms.untracked.string('peak')
    )
)


process.siStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
    appendToDataLabel = cms.string(''),
    PrintDebugOutput = cms.bool(False),
    ThresholdForReducedGranularity = cms.double(0.3),
    UseEmptyRunInfo = cms.bool(False),
    ReduceGranularity = cms.bool(False),
    ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string('SiStripDetVOffRcd'),
        tag = cms.string('')
    ), 
        cms.PSet(
            record = cms.string('SiStripDetCablingRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('RunInfoRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('SiStripBadChannelRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('SiStripBadFiberRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('SiStripBadModuleRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('SiStripBadStripRcd'),
            tag = cms.string('')
        ))
)


process.sistripconn = cms.ESProducer("SiStripConnectivity")


process.templates = cms.ESProducer("PixelCPETemplateRecoESProducer",
    DoLorentz = cms.bool(False),
    DoCosmics = cms.bool(False),
    LoadTemplatesFromDB = cms.bool(True),
    ComponentName = cms.string('PixelCPETemplateReco'),
    Alpha2Order = cms.bool(True),
    ClusterProbComputationFlag = cms.int32(0),
    speed = cms.int32(-2),
    UseClusterSplitter = cms.bool(False)
)


process.tobTecFlexibleKFFittingSmoother = cms.ESProducer("FlexibleKFFittingSmootherESProducer",
    ComponentName = cms.string('tobTecFlexibleKFFittingSmoother'),
    standardFitter = cms.string('tobTecStepFitterSmoother'),
    looperFitter = cms.string('tobTecStepFitterSmootherForLoopers')
)


process.tobTecStepChi2Est = cms.ESProducer("Chi2ChargeMeasurementEstimatorESProducer",
    minGoodStripCharge = cms.double(2069),
    MaxChi2 = cms.double(16.0),
    nSigma = cms.double(3.0),
    ComponentName = cms.string('tobTecStepChi2Est')
)


process.tobTecStepClusterShapeHitFilter = cms.ESProducer("ClusterShapeHitFilterESProducer",
    ComponentName = cms.string('tobTecStepClusterShapeHitFilter'),
    PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par'),
    doStripShapeCut = cms.bool(False),
    minGoodStripCharge = cms.double(2069)
)


process.tobTecStepFitterSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(30),
    LogPixelProbabilityCut = cms.double(-14.0),
    Fitter = cms.string('tobTecStepRKFitter'),
    MinNumberOfHits = cms.int32(8),
    Smoother = cms.string('tobTecStepRKSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('tobTecStepFitterSmoother'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.tobTecStepFitterSmootherForLoopers = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(30),
    LogPixelProbabilityCut = cms.double(-14.0),
    Fitter = cms.string('tobTecStepRKFitterForLoopers'),
    MinNumberOfHits = cms.int32(8),
    Smoother = cms.string('tobTecStepRKSmootherForLoopers'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('tobTecStepFitterSmootherForLoopers'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)


process.tobTecStepRKTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    ComponentName = cms.string('tobTecStepRKFitter'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    minHits = cms.int32(8)
)


process.tobTecStepRKTrajectoryFitterForLoopers = cms.ESProducer("KFTrajectoryFitterESProducer",
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    ComponentName = cms.string('tobTecStepRKFitterForLoopers'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers'),
    minHits = cms.int32(8)
)


process.tobTecStepRKTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(10.0),
    minHits = cms.int32(8),
    ComponentName = cms.string('tobTecStepRKSmoother'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)


process.tobTecStepRKTrajectorySmootherForLoopers = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(10.0),
    minHits = cms.int32(8),
    ComponentName = cms.string('tobTecStepRKSmootherForLoopers'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)


process.tobTecStepTrajectoryCleanerBySharedHits = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('tobTecStepTrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.09),
    ValidHitBonus = cms.double(5.0),
    ComponentType = cms.string('TrajectoryCleanerBySharedHits'),
    MissingHitPenalty = cms.double(20.0),
    allowSharedFirstHit = cms.bool(True)
)


process.trackerGeometry = cms.ESProducer("TrackerDigiGeometryESModule",
    appendToDataLabel = cms.string(''),
    fromDDD = cms.bool(True),
    trackerGeometryConstants = cms.PSet(
        ROCS_X = cms.int32(0),
        ROCS_Y = cms.int32(0),
        upgradeGeometry = cms.bool(False),
        BIG_PIX_PER_ROC_Y = cms.int32(2),
        BIG_PIX_PER_ROC_X = cms.int32(1),
        ROWS_PER_ROC = cms.int32(80),
        COLS_PER_ROC = cms.int32(52)
    ),
    applyAlignment = cms.bool(True),
    alignmentsLabel = cms.string('')
)


process.trackerNumberingGeometry = cms.ESProducer("TrackerGeometricDetESModule",
    appendToDataLabel = cms.string(''),
    fromDDD = cms.bool(True),
    layerNumberPXB = cms.uint32(16),
    totalBlade = cms.uint32(24)
)


process.trackerTopologyConstants = cms.ESProducer("TrackerTopologyEP",
    tob_rodStartBit = cms.uint32(5),
    tib_str_int_extStartBit = cms.uint32(10),
    tib_layerMask = cms.uint32(7),
    pxf_bladeMask = cms.uint32(63),
    appendToDataLabel = cms.string(''),
    pxb_ladderStartBit = cms.uint32(8),
    pxb_layerStartBit = cms.uint32(16),
    tec_wheelStartBit = cms.uint32(14),
    tib_str_int_extMask = cms.uint32(3),
    tec_ringStartBit = cms.uint32(5),
    tib_moduleStartBit = cms.uint32(2),
    tib_sterMask = cms.uint32(3),
    tid_sideStartBit = cms.uint32(13),
    tid_module_fw_bwStartBit = cms.uint32(7),
    tid_ringMask = cms.uint32(3),
    tob_sterMask = cms.uint32(3),
    tec_petal_fw_bwStartBit = cms.uint32(12),
    tec_ringMask = cms.uint32(7),
    tib_strMask = cms.uint32(63),
    tec_sterMask = cms.uint32(3),
    tec_wheelMask = cms.uint32(15),
    tec_sideStartBit = cms.uint32(18),
    pxb_moduleMask = cms.uint32(63),
    pxf_panelStartBit = cms.uint32(8),
    tid_sideMask = cms.uint32(3),
    tob_moduleMask = cms.uint32(7),
    tid_ringStartBit = cms.uint32(9),
    pxf_sideMask = cms.uint32(3),
    pxb_moduleStartBit = cms.uint32(2),
    pxf_diskStartBit = cms.uint32(16),
    tib_str_fw_bwMask = cms.uint32(3),
    tec_moduleMask = cms.uint32(7),
    tid_sterMask = cms.uint32(3),
    tob_rod_fw_bwMask = cms.uint32(3),
    tob_layerStartBit = cms.uint32(14),
    tec_petal_fw_bwMask = cms.uint32(3),
    tib_strStartBit = cms.uint32(4),
    tec_sterStartBit = cms.uint32(0),
    tid_moduleMask = cms.uint32(31),
    tib_sterStartBit = cms.uint32(0),
    tid_sterStartBit = cms.uint32(0),
    pxf_moduleStartBit = cms.uint32(2),
    pxf_diskMask = cms.uint32(15),
    tob_moduleStartBit = cms.uint32(2),
    tid_wheelStartBit = cms.uint32(11),
    tob_layerMask = cms.uint32(7),
    tid_module_fw_bwMask = cms.uint32(3),
    tob_rod_fw_bwStartBit = cms.uint32(12),
    tec_petalMask = cms.uint32(15),
    pxb_ladderMask = cms.uint32(255),
    tec_moduleStartBit = cms.uint32(2),
    tob_rodMask = cms.uint32(127),
    tec_sideMask = cms.uint32(3),
    pxf_sideStartBit = cms.uint32(23),
    pxb_layerMask = cms.uint32(15),
    tib_layerStartBit = cms.uint32(14),
    pxf_panelMask = cms.uint32(3),
    tib_moduleMask = cms.uint32(3),
    pxf_bladeStartBit = cms.uint32(10),
    tid_wheelMask = cms.uint32(3),
    tob_sterStartBit = cms.uint32(0),
    tid_moduleStartBit = cms.uint32(2),
    tec_petalStartBit = cms.uint32(8),
    tib_str_fw_bwStartBit = cms.uint32(12),
    pxf_moduleMask = cms.uint32(63)
)


process.trajectoryCleanerBySharedHits = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('TrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.19),
    ValidHitBonus = cms.double(5.0),
    ComponentType = cms.string('TrajectoryCleanerBySharedHits'),
    MissingHitPenalty = cms.double(20.0),
    allowSharedFirstHit = cms.bool(True)
)


process.ttrhbwor = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    Matcher = cms.string('Fake'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    PixelCPE = cms.string('Fake'),
    ComponentName = cms.string('WithoutRefit')
)


process.ttrhbwr = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
    PixelCPE = cms.string('PixelCPEGeneric'),
    ComponentName = cms.string('WithTrackAngle')
)


process.GlobalTag = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string(''),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(0),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(60),
        authenticationSystem = cms.untracked.int32(0),
        connectionRetrialPeriod = cms.untracked.int32(10)
    ),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelTemplateDBObjectRcd'),
        tag = cms.string('SiPixelTemplateDBObject_38T_v3_mc'),
        connect = cms.untracked.string('frontier://FrontierProd/CMS_COND_31X_PIXEL')
    )),
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'),
    globaltag = cms.string('DESIGN72_V1::All')
)


process.HepPDTESSource = cms.ESSource("HepPDTESSource",
    pdtFileName = cms.FileInPath('SimGeneral/HepPDTESSource/data/pythiaparticle.tbl')
)


process.XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMother.xml', 
        'Geometry/CMSCommonData/data/cmsTracker.xml', 
        'Geometry/CMSCommonData/data/caloBase.xml', 
        'Geometry/CMSCommonData/data/cmsCalo.xml', 
        'Geometry/CMSCommonData/data/muonBase.xml', 
        'Geometry/CMSCommonData/data/cmsMuon.xml', 
        'Geometry/CMSCommonData/data/mgnt.xml', 
        'Geometry/CMSCommonData/data/beampipe.xml', 
        'Geometry/CMSCommonData/data/cmsBeam.xml', 
        'Geometry/CMSCommonData/data/muonMB.xml', 
        'Geometry/CMSCommonData/data/muonMagnet.xml', 
        'Geometry/TrackerCommonData/data/pixfwdMaterials.xml', 
        'Geometry/TrackerCommonData/data/pixfwdCommon.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq1x2.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq1x5.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq2x3.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq2x4.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq2x5.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPanelBase.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPanel.xml', 
        'Geometry/TrackerCommonData/data/pixfwdBlade.xml', 
        'Geometry/TrackerCommonData/data/pixfwdNipple.xml', 
        'Geometry/TrackerCommonData/data/pixfwdDisk.xml', 
        'Geometry/TrackerCommonData/data/pixfwdCylinder.xml', 
        'Geometry/TrackerCommonData/data/pixfwd.xml', 
        'Geometry/TrackerCommonData/data/pixbarmaterial.xml', 
        'Geometry/TrackerCommonData/data/pixbarladder.xml', 
        'Geometry/TrackerCommonData/data/pixbarladderfull.xml', 
        'Geometry/TrackerCommonData/data/pixbarladderhalf.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer0.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer1.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer2.xml', 
        'Geometry/TrackerCommonData/data/pixbar.xml', 
        'Geometry/TrackerCommonData/data/tibtidcommonmaterial.xml', 
        'Geometry/TrackerCommonData/data/tibmaterial.xml', 
        'Geometry/TrackerCommonData/data/tibmodpar.xml', 
        'Geometry/TrackerCommonData/data/tibmodule0.xml', 
        'Geometry/TrackerCommonData/data/tibmodule0a.xml', 
        'Geometry/TrackerCommonData/data/tibmodule0b.xml', 
        'Geometry/TrackerCommonData/data/tibmodule2.xml', 
        'Geometry/TrackerCommonData/data/tibstringpar.xml', 
        'Geometry/TrackerCommonData/data/tibstring0ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring0lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring0ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring0ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring0.xml', 
        'Geometry/TrackerCommonData/data/tibstring1ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring1lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring1ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring1ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring1.xml', 
        'Geometry/TrackerCommonData/data/tibstring2ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring2lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring2ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring2ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring2.xml', 
        'Geometry/TrackerCommonData/data/tibstring3ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring3lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring3ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring3ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring3.xml', 
        'Geometry/TrackerCommonData/data/tiblayerpar.xml', 
        'Geometry/TrackerCommonData/data/tiblayer0.xml', 
        'Geometry/TrackerCommonData/data/tiblayer1.xml', 
        'Geometry/TrackerCommonData/data/tiblayer2.xml', 
        'Geometry/TrackerCommonData/data/tiblayer3.xml', 
        'Geometry/TrackerCommonData/data/tib.xml', 
        'Geometry/TrackerCommonData/data/tidmaterial.xml', 
        'Geometry/TrackerCommonData/data/tidmodpar.xml', 
        'Geometry/TrackerCommonData/data/tidmodule0.xml', 
        'Geometry/TrackerCommonData/data/tidmodule0r.xml', 
        'Geometry/TrackerCommonData/data/tidmodule0l.xml', 
        'Geometry/TrackerCommonData/data/tidmodule1.xml', 
        'Geometry/TrackerCommonData/data/tidmodule1r.xml', 
        'Geometry/TrackerCommonData/data/tidmodule1l.xml', 
        'Geometry/TrackerCommonData/data/tidmodule2.xml', 
        'Geometry/TrackerCommonData/data/tidringpar.xml', 
        'Geometry/TrackerCommonData/data/tidring0.xml', 
        'Geometry/TrackerCommonData/data/tidring0f.xml', 
        'Geometry/TrackerCommonData/data/tidring0b.xml', 
        'Geometry/TrackerCommonData/data/tidring1.xml', 
        'Geometry/TrackerCommonData/data/tidring1f.xml', 
        'Geometry/TrackerCommonData/data/tidring1b.xml', 
        'Geometry/TrackerCommonData/data/tidring2.xml', 
        'Geometry/TrackerCommonData/data/tid.xml', 
        'Geometry/TrackerCommonData/data/tidf.xml', 
        'Geometry/TrackerCommonData/data/tidb.xml', 
        'Geometry/TrackerCommonData/data/tibtidservices.xml', 
        'Geometry/TrackerCommonData/data/tibtidservicesf.xml', 
        'Geometry/TrackerCommonData/data/tibtidservicesb.xml', 
        'Geometry/TrackerCommonData/data/tobmaterial.xml', 
        'Geometry/TrackerCommonData/data/tobmodpar.xml', 
        'Geometry/TrackerCommonData/data/tobmodule0.xml', 
        'Geometry/TrackerCommonData/data/tobmodule2.xml', 
        'Geometry/TrackerCommonData/data/tobmodule4.xml', 
        'Geometry/TrackerCommonData/data/tobrodpar.xml', 
        'Geometry/TrackerCommonData/data/tobrod0c.xml', 
        'Geometry/TrackerCommonData/data/tobrod0l.xml', 
        'Geometry/TrackerCommonData/data/tobrod0h.xml', 
        'Geometry/TrackerCommonData/data/tobrod0.xml', 
        'Geometry/TrackerCommonData/data/tobrod1l.xml', 
        'Geometry/TrackerCommonData/data/tobrod1h.xml', 
        'Geometry/TrackerCommonData/data/tobrod1.xml', 
        'Geometry/TrackerCommonData/data/tobrod2c.xml', 
        'Geometry/TrackerCommonData/data/tobrod2l.xml', 
        'Geometry/TrackerCommonData/data/tobrod2h.xml', 
        'Geometry/TrackerCommonData/data/tobrod2.xml', 
        'Geometry/TrackerCommonData/data/tobrod3l.xml', 
        'Geometry/TrackerCommonData/data/tobrod3h.xml', 
        'Geometry/TrackerCommonData/data/tobrod3.xml', 
        'Geometry/TrackerCommonData/data/tobrod4c.xml', 
        'Geometry/TrackerCommonData/data/tobrod4l.xml', 
        'Geometry/TrackerCommonData/data/tobrod4h.xml', 
        'Geometry/TrackerCommonData/data/tobrod4.xml', 
        'Geometry/TrackerCommonData/data/tobrod5l.xml', 
        'Geometry/TrackerCommonData/data/tobrod5h.xml', 
        'Geometry/TrackerCommonData/data/tobrod5.xml', 
        'Geometry/TrackerCommonData/data/tob.xml', 
        'Geometry/TrackerCommonData/data/tecmaterial.xml', 
        'Geometry/TrackerCommonData/data/tecmodpar.xml', 
        'Geometry/TrackerCommonData/data/tecmodule0.xml', 
        'Geometry/TrackerCommonData/data/tecmodule0r.xml', 
        'Geometry/TrackerCommonData/data/tecmodule0s.xml', 
        'Geometry/TrackerCommonData/data/tecmodule1.xml', 
        'Geometry/TrackerCommonData/data/tecmodule1r.xml', 
        'Geometry/TrackerCommonData/data/tecmodule1s.xml', 
        'Geometry/TrackerCommonData/data/tecmodule2.xml', 
        'Geometry/TrackerCommonData/data/tecmodule3.xml', 
        'Geometry/TrackerCommonData/data/tecmodule4.xml', 
        'Geometry/TrackerCommonData/data/tecmodule4r.xml', 
        'Geometry/TrackerCommonData/data/tecmodule4s.xml', 
        'Geometry/TrackerCommonData/data/tecmodule5.xml', 
        'Geometry/TrackerCommonData/data/tecmodule6.xml', 
        'Geometry/TrackerCommonData/data/tecpetpar.xml', 
        'Geometry/TrackerCommonData/data/tecring0.xml', 
        'Geometry/TrackerCommonData/data/tecring1.xml', 
        'Geometry/TrackerCommonData/data/tecring2.xml', 
        'Geometry/TrackerCommonData/data/tecring3.xml', 
        'Geometry/TrackerCommonData/data/tecring4.xml', 
        'Geometry/TrackerCommonData/data/tecring5.xml', 
        'Geometry/TrackerCommonData/data/tecring6.xml', 
        'Geometry/TrackerCommonData/data/tecring0f.xml', 
        'Geometry/TrackerCommonData/data/tecring1f.xml', 
        'Geometry/TrackerCommonData/data/tecring2f.xml', 
        'Geometry/TrackerCommonData/data/tecring3f.xml', 
        'Geometry/TrackerCommonData/data/tecring4f.xml', 
        'Geometry/TrackerCommonData/data/tecring5f.xml', 
        'Geometry/TrackerCommonData/data/tecring6f.xml', 
        'Geometry/TrackerCommonData/data/tecring0b.xml', 
        'Geometry/TrackerCommonData/data/tecring1b.xml', 
        'Geometry/TrackerCommonData/data/tecring2b.xml', 
        'Geometry/TrackerCommonData/data/tecring3b.xml', 
        'Geometry/TrackerCommonData/data/tecring4b.xml', 
        'Geometry/TrackerCommonData/data/tecring5b.xml', 
        'Geometry/TrackerCommonData/data/tecring6b.xml', 
        'Geometry/TrackerCommonData/data/tecpetalf.xml', 
        'Geometry/TrackerCommonData/data/tecpetalb.xml', 
        'Geometry/TrackerCommonData/data/tecpetal0.xml', 
        'Geometry/TrackerCommonData/data/tecpetal0f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal0b.xml', 
        'Geometry/TrackerCommonData/data/tecpetal3.xml', 
        'Geometry/TrackerCommonData/data/tecpetal3f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal3b.xml', 
        'Geometry/TrackerCommonData/data/tecpetal6f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal6b.xml', 
        'Geometry/TrackerCommonData/data/tecpetal8f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal8b.xml', 
        'Geometry/TrackerCommonData/data/tecwheel.xml', 
        'Geometry/TrackerCommonData/data/tecwheela.xml', 
        'Geometry/TrackerCommonData/data/tecwheelb.xml', 
        'Geometry/TrackerCommonData/data/tecwheelc.xml', 
        'Geometry/TrackerCommonData/data/tecwheeld.xml', 
        'Geometry/TrackerCommonData/data/tecwheel6.xml', 
        'Geometry/TrackerCommonData/data/tecservices.xml', 
        'Geometry/TrackerCommonData/data/tecbackplate.xml', 
        'Geometry/TrackerCommonData/data/tec.xml', 
        'Geometry/TrackerCommonData/data/trackermaterial.xml', 
        'Geometry/TrackerCommonData/data/tracker.xml', 
        'Geometry/TrackerCommonData/data/trackerpixbar.xml', 
        'Geometry/TrackerCommonData/data/trackerpixfwd.xml', 
        'Geometry/TrackerCommonData/data/trackertibtidservices.xml', 
        'Geometry/TrackerCommonData/data/trackertib.xml', 
        'Geometry/TrackerCommonData/data/trackertid.xml', 
        'Geometry/TrackerCommonData/data/trackertob.xml', 
        'Geometry/TrackerCommonData/data/trackertec.xml', 
        'Geometry/TrackerCommonData/data/trackerbulkhead.xml', 
        'Geometry/TrackerCommonData/data/trackerother.xml', 
        'Geometry/EcalCommonData/data/eregalgo.xml', 
        'Geometry/EcalCommonData/data/ebalgo.xml', 
        'Geometry/EcalCommonData/data/ebcon.xml', 
        'Geometry/EcalCommonData/data/ebrot.xml', 
        'Geometry/EcalCommonData/data/eecon.xml', 
        'Geometry/EcalCommonData/data/eefixed.xml', 
        'Geometry/EcalCommonData/data/eehier.xml', 
        'Geometry/EcalCommonData/data/eealgo.xml', 
        'Geometry/EcalCommonData/data/escon.xml', 
        'Geometry/EcalCommonData/data/esalgo.xml', 
        'Geometry/EcalCommonData/data/eeF.xml', 
        'Geometry/EcalCommonData/data/eeB.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'Geometry/HcalCommonData/data/hcalalgo.xml', 
        'Geometry/HcalCommonData/data/hcalbarrelalgo.xml', 
        'Geometry/HcalCommonData/data/hcalendcapalgo.xml', 
        'Geometry/HcalCommonData/data/hcalouteralgo.xml', 
        'Geometry/HcalCommonData/data/hcalforwardalgo.xml', 
        'Geometry/HcalCommonData/data/average/hcalforwardmaterial.xml', 
        'Geometry/MuonCommonData/data/mbCommon.xml', 
        'Geometry/MuonCommonData/data/mb1.xml', 
        'Geometry/MuonCommonData/data/mb2.xml', 
        'Geometry/MuonCommonData/data/mb3.xml', 
        'Geometry/MuonCommonData/data/mb4.xml', 
        'Geometry/MuonCommonData/data/muonYoke.xml', 
        'Geometry/MuonCommonData/data/mf.xml', 
        'Geometry/ForwardCommonData/data/forward.xml', 
        'Geometry/ForwardCommonData/data/bundle/forwardshield.xml', 
        'Geometry/ForwardCommonData/data/brmrotations.xml', 
        'Geometry/ForwardCommonData/data/brm.xml', 
        'Geometry/ForwardCommonData/data/totemMaterials.xml', 
        'Geometry/ForwardCommonData/data/totemRotations.xml', 
        'Geometry/ForwardCommonData/data/totemt1.xml', 
        'Geometry/ForwardCommonData/data/totemt2.xml', 
        'Geometry/ForwardCommonData/data/ionpump.xml', 
        'Geometry/MuonCommonData/data/muonNumbering.xml', 
        'Geometry/TrackerCommonData/data/trackerStructureTopology.xml', 
        'Geometry/TrackerSimData/data/trackersens.xml', 
        'Geometry/TrackerRecoData/data/trackerRecoMaterial.xml', 
        'Geometry/EcalSimData/data/ecalsens.xml', 
        'Geometry/HcalCommonData/data/hcalsenspmf.xml', 
        'Geometry/HcalSimData/data/hf.xml', 
        'Geometry/HcalSimData/data/hfpmt.xml', 
        'Geometry/HcalSimData/data/hffibrebundle.xml', 
        'Geometry/HcalSimData/data/CaloUtil.xml', 
        'Geometry/MuonSimData/data/muonSens.xml', 
        'Geometry/DTGeometryBuilder/data/dtSpecsFilter.xml', 
        'Geometry/CSCGeometryBuilder/data/cscSpecsFilter.xml', 
        'Geometry/CSCGeometryBuilder/data/cscSpecs.xml', 
        'Geometry/RPCGeometryBuilder/data/RPCSpecs.xml', 
        'Geometry/ForwardCommonData/data/brmsens.xml', 
        'Geometry/HcalSimData/data/HcalProdCuts.xml', 
        'Geometry/EcalSimData/data/EcalProdCuts.xml', 
        'Geometry/EcalSimData/data/ESProdCuts.xml', 
        'Geometry/TrackerSimData/data/trackerProdCuts.xml', 
        'Geometry/TrackerSimData/data/trackerProdCutsBEAM.xml', 
        'Geometry/MuonSimData/data/muonProdCuts.xml', 
        'Geometry/ForwardSimData/data/ForwardShieldProdCuts.xml', 
        'Geometry/CMSCommonData/data/FieldParameters.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


process.eegeom = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('EcalMappingRcd'),
    firstValid = cms.vuint32(1)
)


process.es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    HcalReLabel = cms.PSet(
        RelabelRules = cms.untracked.PSet(
            Eta16 = cms.untracked.vint32(1, 1, 2, 2, 2, 
                2, 2, 2, 2, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3),
            Eta17 = cms.untracked.vint32(1, 1, 2, 2, 3, 
                3, 3, 4, 4, 4, 
                4, 4, 5, 5, 5, 
                5, 5, 5, 5),
            Eta1 = cms.untracked.vint32(1, 2, 2, 2, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3),
            CorrectPhi = cms.untracked.bool(False)
        ),
        RelabelHits = cms.untracked.bool(False)
    ),
    HERecalibration = cms.bool(False),
    toGet = cms.untracked.vstring('GainWidths'),
    GainWidthsForTrigPrims = cms.bool(False),
    HEreCalibCutoff = cms.double(20.0),
    HFRecalibration = cms.bool(False),
    iLumi = cms.double(-1.0),
    hcalTopologyConstants = cms.PSet(
        maxDepthHE = cms.int32(3),
        maxDepthHB = cms.int32(2),
        mode = cms.string('HcalTopologyMode::LHC')
    )
)


process.magfield = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMagneticField.xml', 
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1103l.xml', 
        'Geometry/CMSCommonData/data/materials.xml'),
    rootNodeName = cms.string('cmsMagneticField:MAGF')
)


process.myTrackerAlignment = cms.ESSource("PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    appendToDataLabel = cms.string(''),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TrackerIdealGeometry210_mc')
    )),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string(''),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(0),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(60),
        authenticationSystem = cms.untracked.int32(0),
        connectionRetrialPeriod = cms.untracked.int32(10)
    ),
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X')
)


process.myTrackerAlignmentErr = cms.ESSource("PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    appendToDataLabel = cms.string(''),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentErrorRcd'),
        tag = cms.string('TrackerIdealGeometryErrors210_mc')
    )),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string(''),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(0),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(60),
        authenticationSystem = cms.untracked.int32(0),
        connectionRetrialPeriod = cms.untracked.int32(10)
    ),
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X')
)


process.prefer("myTrackerAlignmentErr")

process.prefer("es_hardcode")

process.prefer("magfield")

process.prefer("myTrackerAlignment")

process.ChargeSignificanceTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('ChargeSignificanceTrajectoryFilter'),
    chargeSignificance = cms.double(-1.0)
)

process.CkfBaseTrajectoryFilter_block = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.9),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(999),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    nSigmaMinPt = cms.double(5.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    minimumNumberOfHits = cms.int32(5),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    chargeSignificance = cms.double(-1.0)
)

process.CkfTrajectoryBuilder = cms.PSet(
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    maxCand = cms.int32(5),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('CkfTrajectoryBuilder'),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.CkfTrajectoryBuilderBeamHalo = cms.PSet(
    propagatorAlong = cms.string('BeamHaloPropagatorAlong'),
    maxCand = cms.int32(5),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('ckfTrajectoryFilterBeamHaloMuon')
    ),
    propagatorOpposite = cms.string('BeamHaloPropagatorOpposite'),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('CkfTrajectoryBuilder'),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.ClusterShapeTrajectoryFilter = cms.PSet(
    cacheSrc = cms.InputTag("siPixelClusterShapeCache"),
    ComponentType = cms.string('ClusterShapeTrajectoryFilter')
)

process.CompositeTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet()
)

process.CondDBCommon = cms.PSet(
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string(''),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(0),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(60),
        authenticationSystem = cms.untracked.int32(0),
        connectionRetrialPeriod = cms.untracked.int32(10)
    ),
    connect = cms.string('protocol://db/schema')
)

process.CondDBSetup = cms.PSet(
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string(''),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(0),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(60),
        authenticationSystem = cms.untracked.int32(0),
        connectionRetrialPeriod = cms.untracked.int32(10)
    )
)

process.DefaultAlgorithms = cms.PSet(
    CutToAvoidSignal = cms.double(2.0),
    slopeY = cms.int32(4),
    slopeX = cms.int32(3),
    PedestalSubtractionFedMode = cms.bool(False),
    Fraction = cms.double(0.2),
    minStripsToFit = cms.uint32(4),
    consecThreshold = cms.uint32(5),
    hitStripThreshold = cms.uint32(40),
    Deviation = cms.uint32(25),
    CommonModeNoiseSubtractionMode = cms.string('IteratedMedian'),
    filteredBaselineDerivativeSumSquare = cms.double(30),
    ApplyBaselineCleaner = cms.bool(True),
    doAPVRestore = cms.bool(True),
    TruncateInSuppressor = cms.bool(True),
    restoreThreshold = cms.double(0.5),
    APVInspectMode = cms.string('BaselineFollower'),
    ForceNoRestore = cms.bool(False),
    useRealMeanCM = cms.bool(False),
    ApplyBaselineRejection = cms.bool(True),
    DeltaCMThreshold = cms.uint32(20),
    nSigmaNoiseDerTh = cms.uint32(4),
    nSaturatedStrip = cms.uint32(2),
    SiStripFedZeroSuppressionMode = cms.uint32(4),
    useCMMeanMap = cms.bool(False),
    SelfSelectRestoreAlgo = cms.bool(False),
    distortionThreshold = cms.uint32(20),
    filteredBaselineMax = cms.double(6),
    Iterations = cms.int32(3),
    CleaningSequence = cms.uint32(1),
    nSmooth = cms.uint32(9),
    APVRestoreMode = cms.string('BaselineFollower'),
    MeanCM = cms.int32(0)
)

process.DefaultClusterizer = cms.PSet(
    ChannelThreshold = cms.double(2.0),
    MaxSequentialBad = cms.uint32(1),
    Algorithm = cms.string('ThreeThresholdAlgorithm'),
    MaxSequentialHoles = cms.uint32(0),
    MaxAdjacentBad = cms.uint32(0),
    QualityLabel = cms.string(''),
    SeedThreshold = cms.double(3.0),
    RemoveApvShots = cms.bool(True),
    ClusterThreshold = cms.double(5.0)
)

process.GroupedCkfTrajectoryBuilder = cms.PSet(
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    maxCand = cms.int32(5),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    lostHitPenalty = cms.double(30.0),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    estimator = cms.string('Chi2'),
    intermediateCleaning = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(5)
)

process.GroupedCkfTrajectoryBuilderP5 = cms.PSet(
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    maxCand = cms.int32(1),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('ckfBaseTrajectoryFilterP5')
    ),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    alwaysUseInvalidHits = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(5),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    estimator = cms.string('Chi2'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.GroupedCkfTrajectoryBuilderP5Bottom = cms.PSet(
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    maxCand = cms.int32(1),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('ckfBaseTrajectoryFilterP5')
    ),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    alwaysUseInvalidHits = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(5),
    MeasurementTrackerName = cms.string('MeasurementTrackerBottom'),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    estimator = cms.string('Chi2'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.GroupedCkfTrajectoryBuilderP5Top = cms.PSet(
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    maxCand = cms.int32(1),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('ckfBaseTrajectoryFilterP5')
    ),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    alwaysUseInvalidHits = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(5),
    MeasurementTrackerName = cms.string('MeasurementTrackerTop'),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    estimator = cms.string('Chi2'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.HcalReLabel = cms.PSet(
    RelabelRules = cms.untracked.PSet(
        Eta16 = cms.untracked.vint32(1, 1, 2, 2, 2, 
            2, 2, 2, 2, 3, 
            3, 3, 3, 3, 3, 
            3, 3, 3, 3),
        Eta17 = cms.untracked.vint32(1, 1, 2, 2, 3, 
            3, 3, 4, 4, 4, 
            4, 4, 5, 5, 5, 
            5, 5, 5, 5),
        Eta1 = cms.untracked.vint32(1, 2, 2, 2, 3, 
            3, 3, 3, 3, 3, 
            3, 3, 3, 3, 3, 
            3, 3, 3, 3),
        CorrectPhi = cms.untracked.bool(False)
    ),
    RelabelHits = cms.untracked.bool(False)
)

process.MaxConsecLostHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MaxConsecLostHitsTrajectoryFilter'),
    maxConsecLostHits = cms.int32(1)
)

process.MaxHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MaxHitsTrajectoryFilter'),
    maxNumberOfHits = cms.int32(100)
)

process.MaxLostHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MaxLostHitsTrajectoryFilter'),
    maxLostHits = cms.int32(1)
)

process.MinHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MinHitsTrajectoryFilter'),
    minimumNumberOfHits = cms.int32(5)
)

process.MinPtTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MinPtTrajectoryFilter'),
    nSigmaMinPt = cms.double(5.0),
    minHitsMinPt = cms.int32(3),
    minPt = cms.double(1.0)
)

process.PixelTripletHLTGenerator = cms.PSet(
    useBending = cms.bool(True),
    useFixedPreFiltering = cms.bool(False),
    maxElement = cms.uint32(100000),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    extraHitRPhitolerance = cms.double(0.032),
    useMultScattering = cms.bool(True),
    phiPreFiltering = cms.double(0.3),
    extraHitRZtolerance = cms.double(0.037),
    ComponentName = cms.string('PixelTripletHLTGenerator')
)

process.PixelTripletHLTGeneratorWithFilter = cms.PSet(
    useBending = cms.bool(True),
    useFixedPreFiltering = cms.bool(False),
    maxElement = cms.uint32(100000),
    SeedComparitorPSet = cms.PSet(
        clusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
        ComponentName = cms.string('LowPtClusterShapeSeedComparitor')
    ),
    extraHitRPhitolerance = cms.double(0.032),
    useMultScattering = cms.bool(True),
    phiPreFiltering = cms.double(0.3),
    extraHitRZtolerance = cms.double(0.037),
    ComponentName = cms.string('PixelTripletHLTGenerator')
)

process.PixelTripletLargeTipGenerator = cms.PSet(
    useBending = cms.bool(True),
    useFixedPreFiltering = cms.bool(False),
    maxElement = cms.uint32(100000),
    ComponentName = cms.string('PixelTripletLargeTipGenerator'),
    extraHitRPhitolerance = cms.double(0.0),
    useMultScattering = cms.bool(True),
    phiPreFiltering = cms.double(0.3),
    extraHitRZtolerance = cms.double(0.0)
)

process.RegionPSetBlock = cms.PSet(
    RegionPSet = cms.PSet(
        precise = cms.bool(True),
        originHalfLength = cms.double(21.2),
        originZPos = cms.double(0.0),
        originYPos = cms.double(0.0),
        ptMin = cms.double(0.9),
        originXPos = cms.double(0.0),
        originRadius = cms.double(0.2)
    )
)

process.RegionPSetWithVerticesBlock = cms.PSet(
    RegionPSet = cms.PSet(
        precise = cms.bool(True),
        useFakeVertices = cms.bool(False),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        useFixedError = cms.bool(True),
        originRadius = cms.double(0.2),
        sigmaZVertex = cms.double(3.0),
        fixedError = cms.double(0.2),
        VertexCollection = cms.InputTag("pixelVertices"),
        ptMin = cms.double(0.9),
        useFoundVertices = cms.bool(True),
        nSigmaZ = cms.double(4.0)
    )
)

process.RegionPsetFomBeamSpotBlock = cms.PSet(
    RegionPSet = cms.PSet(
        precise = cms.bool(True),
        nSigmaZ = cms.double(4.0),
        originRadius = cms.double(0.2),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        ptMin = cms.double(0.9)
    )
)

process.SiPixelGainCalibrationServiceParameters = cms.PSet(

)

process.TECi = cms.PSet(
    minRing = cms.int32(1),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    useRingSlector = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    maxRing = cms.int32(2)
)

process.ThresholdPtTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('ThresholdPtTrajectoryFilter'),
    nSigmaThresholdPt = cms.double(5.0),
    minHitsThresholdPt = cms.int32(3),
    thresholdPt = cms.double(10.0)
)

process.ckfBaseInOutTrajectoryFilter = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.9),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(999),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(5)
)

process.ckfBaseTrajectoryFilterP5 = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.5),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(4),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(3),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(5)
)

process.ckfTrajectoryFilterBeamHaloMuon = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.1),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(3),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(2),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(4)
)

process.conv2CkfTrajectoryBuilder = cms.PSet(
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    minNrOfHitsForRebuild = cms.int32(3),
    maxCand = cms.int32(2),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('conv2CkfTrajectoryFilter')
    ),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    alwaysUseInvalidHits = cms.bool(True),
    clustersToSkip = cms.InputTag("conv2Clusters"),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    estimator = cms.string('Chi2'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.conv2CkfTrajectoryFilter = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.1),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(1),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(3)
)

process.convCkfTrajectoryBuilder = cms.PSet(
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    maxCand = cms.int32(1),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('convCkfTrajectoryFilter')
    ),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    alwaysUseInvalidHits = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(3),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    estimator = cms.string('Chi2'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.convCkfTrajectoryFilter = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.1),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(1),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(3)
)

process.detachedTripletStepTrajectoryBuilder = cms.PSet(
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    maxPtForLooperReconstruction = cms.double(0.7),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    lockHits = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    bestHitOnly = cms.bool(True),
    maxCand = cms.int32(3),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('detachedTripletStepTrajectoryFilter')
    ),
    alwaysUseInvalidHits = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(5),
    intermediateCleaning = cms.bool(True),
    estimator = cms.string('detachedTripletStepChi2Est'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.detachedTripletStepTrajectoryFilter = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.075),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(999),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    constantValueForLostHitsFractionFilter = cms.double(0.701),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(3)
)

process.fieldScaling = cms.PSet(
    scalingVolumes = cms.vint32(14100, 14200, 17600, 17800, 17900, 
        18100, 18300, 18400, 18600, 23100, 
        23300, 23400, 23600, 23800, 23900, 
        24100, 28600, 28800, 28900, 29100, 
        29300, 29400, 29600, 28609, 28809, 
        28909, 29109, 29309, 29409, 29609, 
        28610, 28810, 28910, 29110, 29310, 
        29410, 29610, 28611, 28811, 28911, 
        29111, 29311, 29411, 29611),
    scalingFactors = cms.vdouble(1, 1, 0.994, 1.004, 1.004, 
        1.005, 1.004, 1.004, 0.994, 0.965, 
        0.958, 0.958, 0.953, 0.958, 0.958, 
        0.965, 0.918, 0.924, 0.924, 0.906, 
        0.924, 0.924, 0.918, 0.991, 0.998, 
        0.998, 0.978, 0.998, 0.998, 0.991, 
        0.991, 0.998, 0.998, 0.978, 0.998, 
        0.998, 0.991, 0.991, 0.998, 0.998, 
        0.978, 0.998, 0.998, 0.991)
)

process.initialStepTrajectoryBuilder = cms.PSet(
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    maxPtForLooperReconstruction = cms.double(0.7),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    lockHits = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    bestHitOnly = cms.bool(True),
    maxCand = cms.int32(3),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('initialStepTrajectoryFilter')
    ),
    alwaysUseInvalidHits = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(5),
    intermediateCleaning = cms.bool(True),
    estimator = cms.string('initialStepChi2Est'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.initialStepTrajectoryFilter = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.2),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(999),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(3)
)

process.jetCoreRegionalStepTrajectoryBuilder = cms.PSet(
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    maxPtForLooperReconstruction = cms.double(0.7),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    lockHits = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    bestHitOnly = cms.bool(True),
    maxCand = cms.int32(50),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('jetCoreRegionalStepTrajectoryFilter')
    ),
    alwaysUseInvalidHits = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(5),
    intermediateCleaning = cms.bool(True),
    estimator = cms.string('jetCoreRegionalStepChi2Est'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.jetCoreRegionalStepTrajectoryFilter = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.1),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(999),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(3)
)

process.layerInfo = cms.PSet(
    TEC4 = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    TEC5 = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    TEC6 = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    TEC = cms.PSet(
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(7)
    ),
    TEC1 = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    TEC2 = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    TEC3 = cms.PSet(
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits')
    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    )
)

process.lowPtTripletStepStandardTrajectoryFilter = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.075),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(999),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(3)
)

process.lowPtTripletStepTrajectoryBuilder = cms.PSet(
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    maxPtForLooperReconstruction = cms.double(0.7),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    lockHits = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    bestHitOnly = cms.bool(True),
    maxCand = cms.int32(4),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('lowPtTripletStepTrajectoryFilter')
    ),
    alwaysUseInvalidHits = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(5),
    intermediateCleaning = cms.bool(True),
    estimator = cms.string('lowPtTripletStepChi2Est'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.lowPtTripletStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(cms.PSet(
        refToPSet_ = cms.string('lowPtTripletStepStandardTrajectoryFilter')
    ), 
        cms.PSet(
            refToPSet_ = cms.string('ClusterShapeTrajectoryFilter')
        ))
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.mixedTripletStepTrajectoryBuilder = cms.PSet(
    propagatorAlong = cms.string('mixedTripletStepPropagator'),
    propagatorOpposite = cms.string('mixedTripletStepPropagatorOpposite'),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    maxPtForLooperReconstruction = cms.double(0.7),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    lockHits = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    bestHitOnly = cms.bool(True),
    maxCand = cms.int32(2),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('mixedTripletStepTrajectoryFilter')
    ),
    alwaysUseInvalidHits = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(5),
    intermediateCleaning = cms.bool(True),
    estimator = cms.string('mixedTripletStepChi2Est'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.mixedTripletStepTrajectoryFilter = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.1),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(0),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(3)
)

process.muonSeededTrajectoryBuilderForInOut = cms.PSet(
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    maxCand = cms.int32(5),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('muonSeededTrajectoryFilterForInOut')
    ),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    alwaysUseInvalidHits = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(2),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(1000.0),
    updator = cms.string('KFUpdator'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('muonSeededTrajectoryFilterForInOut')
    ),
    keepOriginalIfRebuildFails = cms.bool(True),
    requireSeedHitsInRebuild = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    estimator = cms.string('muonSeededMeasurementEstimatorForInOut'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(1.0)
)

process.muonSeededTrajectoryBuilderForOutIn = cms.PSet(
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    maxCand = cms.int32(3),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('muonSeededTrajectoryFilterForOutIn')
    ),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    alwaysUseInvalidHits = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(5),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(1000.0),
    updator = cms.string('KFUpdator'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('muonSeededTrajectoryFilterForOutIn')
    ),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    estimator = cms.string('muonSeededMeasurementEstimatorForOutIn'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(1.0)
)

process.muonSeededTrajectoryFilterForInOut = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.9),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(999),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    constantValueForLostHitsFractionFilter = cms.double(10),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(3)
)

process.muonSeededTrajectoryFilterForOutIn = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.9),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(999),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    nSigmaMinPt = cms.double(5.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    minimumNumberOfHits = cms.int32(5),
    constantValueForLostHitsFractionFilter = cms.double(10),
    chargeSignificance = cms.double(-1.0)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.pixelLessStepTrajectoryBuilder = cms.PSet(
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    maxPtForLooperReconstruction = cms.double(0.7),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    lockHits = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    bestHitOnly = cms.bool(True),
    maxCand = cms.int32(2),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('pixelLessStepTrajectoryFilter')
    ),
    alwaysUseInvalidHits = cms.bool(False),
    minNrOfHitsForRebuild = cms.int32(4),
    intermediateCleaning = cms.bool(True),
    estimator = cms.string('pixelLessStepChi2Est'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.pixelLessStepTrajectoryFilter = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.1),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(0),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(4)
)

process.pixelPairStepTrajectoryBuilder = cms.PSet(
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    maxPtForLooperReconstruction = cms.double(0.7),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    lockHits = cms.bool(True),
    useSameTrajFilter = cms.bool(True),
    bestHitOnly = cms.bool(True),
    maxCand = cms.int32(3),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('pixelPairStepTrajectoryFilter')
    ),
    alwaysUseInvalidHits = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(5),
    intermediateCleaning = cms.bool(True),
    estimator = cms.string('pixelPairStepChi2Est'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.pixelPairStepTrajectoryFilter = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.1),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(999),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(3)
)

process.tobTecStepInOutTrajectoryFilter = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.1),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(0),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    nSigmaMinPt = cms.double(5.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    minimumNumberOfHits = cms.int32(4),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    chargeSignificance = cms.double(-1.0)
)

process.tobTecStepTrajectoryBuilder = cms.PSet(
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('tobTecStepInOutTrajectoryFilter')
    ),
    maxPtForLooperReconstruction = cms.double(0.7),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    lockHits = cms.bool(True),
    useSameTrajFilter = cms.bool(False),
    bestHitOnly = cms.bool(True),
    maxCand = cms.int32(2),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('tobTecStepTrajectoryFilter')
    ),
    alwaysUseInvalidHits = cms.bool(False),
    minNrOfHitsForRebuild = cms.int32(4),
    intermediateCleaning = cms.bool(True),
    estimator = cms.string('tobTecStepChi2Est'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    keepOriginalIfRebuildFails = cms.bool(False),
    requireSeedHitsInRebuild = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

process.tobTecStepTrajectoryFilter = cms.PSet(
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
    minPt = cms.double(0.1),
    minNumberOfHits = cms.int32(13),
    minHitsMinPt = cms.int32(3),
    maxLostHitsFraction = cms.double(0.1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(0),
    maxNumberOfHits = cms.int32(100),
    maxConsecLostHits = cms.int32(1),
    constantValueForLostHitsFractionFilter = cms.double(1.0),
    minNumberOfHitsPerLoop = cms.int32(4),
    chargeSignificance = cms.double(-1.0),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(6)
)


