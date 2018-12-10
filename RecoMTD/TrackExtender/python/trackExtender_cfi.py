import FWCore.ParameterSet.Config as cms

trackExtender = cms.EDProducer(
    'TrackExtenderWithMTD',
    tracksSrc = cms.InputTag("generalTracks"),
    hitsSrc = cms.InputTag("mtdTrackingRecHits"),
    beamSpotSrc = cms.InputTag("offlineBeamSpot"),
    updateTrackTrajectory = cms.bool(False),
    updateTrackExtra = cms.bool(True),
    updateTrackHitPattern = cms.bool(True),
    TrackTransformer = cms.PSet(
        DoPredictionsOnly = cms.bool(False),
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        #TrackerRecHitBuilder = cms.string('WithTrackAngleAndTemplate'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        MTDRecHitBuilder = cms.string('MTDRecHitBuilder'),
        RefitDirection = cms.string('alongMomentum'),
        RefitRPCHits = cms.bool(True),
        Propagator = cms.string('SmartPropagatorAnyRKOpposite')
        )
    )
