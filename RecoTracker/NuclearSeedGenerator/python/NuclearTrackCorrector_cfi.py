import FWCore.ParameterSet.Config as cms

TrackCorrector = cms.EDProducer("NuclearTrackCorrector",
    InputTrajectory = cms.string('TrackRefitter'),
    KeepOnlyCorrectedTracks = cms.int32(0),
    InputNuclearSeed = cms.string('nuclearSeed'),
    # necessary for the fitting
    Fitter = cms.string('KFFittingSmoother'),
    # necessary for the TrackProducerAlgorithm
    useHitsSplitting = cms.bool(False),
    Verbosity = cms.int32(0),
    InputHitDistance = cms.int32(3),
    Propagator = cms.string('PropagatorWithMaterial'),
    # nested parameter set for TransientInitialStateEstimator
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)


