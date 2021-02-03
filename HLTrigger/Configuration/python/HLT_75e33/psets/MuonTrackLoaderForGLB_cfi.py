import FWCore.ParameterSet.Config as cms

MuonTrackLoaderForGLB = cms.PSet(
    TrackLoaderParameters = cms.PSet(
        DoSmoothing = cms.bool(True),
        MuonUpdatorAtVertexParameters = cms.PSet(
            BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3),
            MaxChi2 = cms.double(1000000.0),
            Propagator = cms.string('SteppingHelixPropagatorOpposite')
        ),
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        TTRHBuilder = cms.string('WithAngleAndTemplate'),
        VertexConstraint = cms.bool(False),
        beamSpot = cms.InputTag("offlineBeamSpot")
    )
)