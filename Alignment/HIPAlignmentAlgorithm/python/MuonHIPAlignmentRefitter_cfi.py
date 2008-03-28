import FWCore.ParameterSet.Config as cms

MuonHIPAlignmentRefitter = cms.EDFilter("MuonHIPAlignmentRefitter",
    MuonPropagator = cms.string('SmartPropagatorAnyOpposite'),
    MuonSource = cms.InputTag("ALCARECOMuAlZMuMu","SelectedMuons"),
    TrackerTrackTransformer = cms.PSet(
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        TrackerRecHitBuilder = cms.string('WithoutRefit'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        RefitDirection = cms.string('insideOut'),
        RefitRPCHits = cms.bool(False),
        Propagator = cms.string('SmartPropagatorAnyOpposite')
    )
)


