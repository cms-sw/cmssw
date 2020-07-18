import FWCore.ParameterSet.Config as cms

hltPhase2L1TrackSeedsFromL1Tracks = cms.EDProducer(
    "SeedGeneratorFromL1TTracksEDProducer",
    InputCollection=cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"),
    MeasurementTrackerEvent=cms.InputTag("MeasurementTrackerEvent"),
    errorSFHitless=cms.double(1e-09),
    estimator=cms.string("hltPhase2L1TrackStepChi2Est"),
    maxEtaForTOB=cms.double(1.2),
    minEtaForTEC=cms.double(0.9),
    propagator=cms.string("PropagatorWithMaterial"),
)

hltPhase2L1TrackCandidates = cms.EDProducer(
    "CkfTrackCandidateMaker",
    MeasurementTrackerEvent=cms.InputTag("MeasurementTrackerEvent"),
    NavigationSchool=cms.string("SimpleNavigationSchool"),
    RedundantSeedCleaner=cms.string("CachingSeedCleanerBySharedInput"),
    SimpleMagneticField=cms.string(""),
    TrajectoryBuilder=cms.string("GroupedCkfTrajectoryBuilder"),
    TrajectoryBuilderPSet=cms.PSet(
        refToPSet_=cms.string("hltPhase2L1TrackStepTrajectoryBuilder")
    ),
    TrajectoryCleaner=cms.string("TrajectoryCleanerBySharedHits"),
    TransientInitialStateEstimatorParameters=cms.PSet(
        numberMeasurementsForFit=cms.int32(4),
        propagatorAlongTISE=cms.string("PropagatorWithMaterial"),
        propagatorOppositeTISE=cms.string("PropagatorWithMaterialOpposite"),
    ),
    alias=cms.string("hltPhase2L1TrackCandidates"),
    cleanTrajectoryAfterInOut=cms.bool(False),
    doSeedingRegionRebuilding=cms.bool(True),
    maxNSeeds=cms.uint32(500000),
    maxSeedsBeforeCleaning=cms.uint32(10000),
    reverseTrajectories=cms.bool(False),
    src=cms.InputTag("hltPhase2L1TrackSeedsFromL1Tracks"),
    useHitsSplitting=cms.bool(True),
)


hltPhase2L1CtfTracks = cms.EDProducer(
    "TrackProducer",
    AlgorithmName=cms.string("hltIter0"),
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
    src=cms.InputTag("hltPhase2L1TrackCandidates"),
    useHitsSplitting=cms.bool(False),
    useSimpleMF=cms.bool(False),
)

hltPhase2L1TracksSeqPattern = cms.Sequence(
    hltPhase2L1TrackSeedsFromL1Tracks
    + hltPhase2L1TrackCandidates
    + hltPhase2L1CtfTracks
)
