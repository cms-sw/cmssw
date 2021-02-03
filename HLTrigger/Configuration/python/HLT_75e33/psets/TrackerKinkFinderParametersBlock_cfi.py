import FWCore.ParameterSet.Config as cms

TrackerKinkFinderParametersBlock = cms.PSet(
    TrackerKinkFinderParameters = cms.PSet(
        DoPredictionsOnly = cms.bool(False),
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        MTDRecHitBuilder = cms.string('MTDRecHitBuilder'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        Propagator = cms.string('SmartPropagatorAnyRKOpposite'),
        RefitDirection = cms.string('alongMomentum'),
        RefitRPCHits = cms.bool(True),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        TrackerRecHitBuilder = cms.string('WithAngleAndTemplate'),
        diagonalOnly = cms.bool(False),
        usePosition = cms.bool(True)
    )
)