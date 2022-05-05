import FWCore.ParameterSet.Config as cms

hltEgammaGsfTracksUnseeded = cms.EDProducer("GsfTrackProducer",
    AlgorithmName = cms.string('gsf'),
    Fitter = cms.string('GsfElectronFittingSmoother'),
    GeometricInnerState = cms.bool(False),
    MeasurementTracker = cms.string(''),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    Propagator = cms.string('fwdGsfElectronPropagator'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    TrajectoryInEvent = cms.bool(False),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    producer = cms.string(''),
    src = cms.InputTag("hltEgammaCkfTrackCandidatesForGSFUnseeded"),
    useHitsSplitting = cms.bool(False)
)
