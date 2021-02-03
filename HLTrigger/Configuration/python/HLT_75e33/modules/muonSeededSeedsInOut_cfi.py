import FWCore.ParameterSet.Config as cms

muonSeededSeedsInOut = cms.EDProducer("MuonReSeeder",
    DoPredictionsOnly = cms.bool(False),
    Fitter = cms.string('KFFitterForRefitInsideOut'),
    MTDRecHitBuilder = cms.string('MTDRecHitBuilder'),
    MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
    Propagator = cms.string('SmartPropagatorAnyRKOpposite'),
    RefitDirection = cms.string('alongMomentum'),
    RefitRPCHits = cms.bool(True),
    Smoother = cms.string('KFSmootherForRefitInsideOut'),
    TrackerRecHitBuilder = cms.string('WithAngleAndTemplate'),
    cut = cms.string('pt > 2'),
    debug = cms.untracked.bool(False),
    insideOut = cms.bool(True),
    layersToKeep = cms.int32(5),
    src = cms.InputTag("earlyMuons")
)
