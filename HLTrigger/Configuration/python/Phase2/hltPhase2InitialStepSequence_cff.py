import FWCore.ParameterSet.Config as cms

hltPhase2InitialStepSeeds = cms.EDProducer(
    "SeedGeneratorFromProtoTracksEDProducer",
    InputCollection=cms.InputTag("hltPhase2PixelTracks"),
    InputVertexCollection=cms.InputTag(""),
    SeedCreatorPSet=cms.PSet(refToPSet_=cms.string("hltPhase2SeedFromProtoTracks")),
    TTRHBuilder=cms.string("WithTrackAngle"),
    originHalfLength=cms.double(0.3),
    originRadius=cms.double(0.1),
    useEventsWithNoVertex=cms.bool(True),
    usePV=cms.bool(False),
    useProtoTrackKinematics=cms.bool(False),
)

hltPhase2InitialStepTrackCandidates = cms.EDProducer(
    "CkfTrackCandidateMaker",
    MeasurementTrackerEvent=cms.InputTag("MeasurementTrackerEvent"),
    NavigationSchool=cms.string("SimpleNavigationSchool"),
    RedundantSeedCleaner=cms.string("CachingSeedCleanerBySharedInput"),
    SimpleMagneticField=cms.string("ParabolicMf"),
    TrajectoryBuilder=cms.string("GroupedCkfTrajectoryBuilder"),
    TrajectoryBuilderPSet=cms.PSet(
        refToPSet_=cms.string("hltPhase2InitialStepTrajectoryBuilder")
    ),
    TrajectoryCleaner=cms.string("TrajectoryCleanerBySharedHits"),
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
    reverseTrajectories=cms.bool(False),
    src=cms.InputTag("hltPhase2InitialStepSeeds"),
    useHitsSplitting=cms.bool(False),
)

hltPhase2InitialStepTracks = cms.EDProducer(
    "TrackProducer",
    AlgorithmName=cms.string("initialStep"),
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
    src=cms.InputTag("hltPhase2InitialStepTrackCandidates"),
    useHitsSplitting=cms.bool(False),
    useSimpleMF=cms.bool(False),
)

hltPhase2InitialStepTrackCutClassifier = cms.EDProducer(
    "TrackCutClassifier",
    beamspot=cms.InputTag("offlineBeamSpot"),
    ignoreVertices=cms.bool(False),
    mva=cms.PSet(
        dr_par=cms.PSet(
            d0err=cms.vdouble(0.003, 0.003, 0.003),
            d0err_par=cms.vdouble(0.001, 0.001, 0.001),
            dr_exp=cms.vint32(4, 4, 4),
            dr_par1=cms.vdouble(0.8, 0.7, 0.6),
            dr_par2=cms.vdouble(0.6, 0.5, 0.45),
        ),
        dz_par=cms.PSet(
            dz_exp=cms.vint32(4, 4, 4),
            dz_par1=cms.vdouble(0.9, 0.8, 0.7),
            dz_par2=cms.vdouble(0.8, 0.7, 0.55),
        ),
        maxChi2=cms.vdouble(9999.0, 25.0, 16.0),
        maxChi2n=cms.vdouble(2.0, 1.4, 1.2),
        maxDr=cms.vdouble(0.5, 0.03, 3.40282346639e38),
        maxDz=cms.vdouble(0.5, 0.2, 3.40282346639e38),
        maxDzWrtBS=cms.vdouble(3.40282346639e38, 24.0, 15.0),
        maxLostLayers=cms.vint32(3, 2, 2),
        min3DLayers=cms.vint32(3, 3, 3),
        minLayers=cms.vint32(3, 3, 3),
        minNVtxTrk=cms.int32(3),
        minNdof=cms.vdouble(1e-05, 1e-05, 1e-05),
        minPixelHits=cms.vint32(0, 0, 3),
    ),
    qualityCuts=cms.vdouble(-0.7, 0.1, 0.7),
    src=cms.InputTag("hltPhase2InitialStepTracks"),
    vertices=cms.InputTag("hltPhase2PixelVertices"),
)

hltPhase2InitialStepTrackSelectionHighPurity = cms.EDProducer(
    "TrackCollectionFilterCloner",
    copyExtras=cms.untracked.bool(True),
    copyTrajectories=cms.untracked.bool(False),
    minQuality=cms.string("highPurity"),
    originalMVAVals=cms.InputTag("hltPhase2InitialStepTrackCutClassifier", "MVAValues"),
    originalQualVals=cms.InputTag(
        "hltPhase2InitialStepTrackCutClassifier", "QualityMasks"
    ),
    originalSource=cms.InputTag("hltPhase2InitialStepTracks"),
)

hltPhase2InitialStepSequence = cms.Sequence(
    hltPhase2InitialStepSeeds
    + hltPhase2InitialStepTrackCandidates
    + hltPhase2InitialStepTracks
    + hltPhase2InitialStepTrackCutClassifier
    + hltPhase2InitialStepTrackSelectionHighPurity
)
