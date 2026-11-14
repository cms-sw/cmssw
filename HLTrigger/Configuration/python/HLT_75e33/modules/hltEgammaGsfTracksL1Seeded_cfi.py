import FWCore.ParameterSet.Config as cms

hltEgammaGsfTracksL1Seeded = cms.EDProducer("GsfTrackProducer",
    AlgorithmName = cms.string('gsf'),
    Fitter = cms.string('hltESPGsfElectronFittingSmoother'),
    GeometricInnerState = cms.bool(False),
    MeasurementTracker = cms.string('hltESPMeasurementTracker'),
    MeasurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    Propagator = cms.string('hltESPFwdElectronPropagator'),
    TTRHBuilder = cms.string('hltESPTTRHBuilderWithTrackAngle'),
    TrajectoryInEvent = cms.bool(False),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    src = cms.InputTag("hltEgammaCkfTrackCandidatesForGSFL1Seeded"),
    useHitsSplitting = cms.bool(False)
)
