import FWCore.ParameterSet.Config as cms

hltEgammaRegionalCTFFinalFitWithMaterial = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltEgammaRegionalCkfTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('hltEgammaRegionalCTFFinalFitWithMaterial'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


