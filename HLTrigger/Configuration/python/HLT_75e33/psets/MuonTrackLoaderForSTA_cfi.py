import FWCore.ParameterSet.Config as cms

MuonTrackLoaderForSTA = cms.PSet(
    TrackLoaderParameters = cms.PSet(
        DoSmoothing = cms.bool(False),
        MuonUpdatorAtVertexParameters = cms.PSet(
            BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3),
            MaxChi2 = cms.double(1000000.0),
            Propagator = cms.string('SteppingHelixPropagatorOpposite')
        ),
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        TTRHBuilder = cms.string('WithAngleAndTemplate'),
        VertexConstraint = cms.bool(True),
        beamSpot = cms.InputTag("offlineBeamSpot")
    )
)