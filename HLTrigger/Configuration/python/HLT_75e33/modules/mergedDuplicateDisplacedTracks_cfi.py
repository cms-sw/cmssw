import FWCore.ParameterSet.Config as cms

mergedDuplicateDisplacedTracks = cms.EDProducer("TrackProducer",
    AlgorithmName = cms.string('undefAlgorithm'),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    GeometricInnerState = cms.bool(False),
    MeasurementTracker = cms.string(''),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    SimpleMagneticField = cms.string(''),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    TrajectoryInEvent = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    clusterRemovalInfo = cms.InputTag(""),
    src = cms.InputTag("duplicateDisplacedTrackCandidates","candidates"),
    useHitsSplitting = cms.bool(False),
    useSimpleMF = cms.bool(False)
)
