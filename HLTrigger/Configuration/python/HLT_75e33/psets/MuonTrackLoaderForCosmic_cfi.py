import FWCore.ParameterSet.Config as cms

MuonTrackLoaderForCosmic = cms.PSet(
    TrackLoaderParameters = cms.PSet(
        AllowNoVertex = cms.untracked.bool(True),
        DoSmoothing = cms.bool(False),
        MuonUpdatorAtVertexParameters = cms.PSet(
            BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3),
            MaxChi2 = cms.double(1000000.0),
            Propagator = cms.string('SteppingHelixPropagatorAny')
        ),
        PutTrajectoryIntoEvent = cms.untracked.bool(False),
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        TTRHBuilder = cms.string('WithAngleAndTemplate'),
        VertexConstraint = cms.bool(False),
        beamSpot = cms.InputTag("offlineBeamSpot")
    )
)