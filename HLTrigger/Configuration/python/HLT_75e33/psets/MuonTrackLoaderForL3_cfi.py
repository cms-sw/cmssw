import FWCore.ParameterSet.Config as cms

MuonTrackLoaderForL3 = cms.PSet(
    TrackLoaderParameters = cms.PSet(
        DoSmoothing = cms.bool(True),
        MuonSeededTracksInstance = cms.untracked.string('L2Seeded'),
        MuonUpdatorAtVertexParameters = cms.PSet(
            BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3),
            MaxChi2 = cms.double(1000000.0),
            Propagator = cms.string('SteppingHelixPropagatorOpposite')
        ),
        PutTkTrackIntoEvent = cms.untracked.bool(True),
        SmoothTkTrack = cms.untracked.bool(False),
        Smoother = cms.string('KFSmootherForMuonTrackLoaderL3'),
        TTRHBuilder = cms.string('WithAngleAndTemplate'),
        VertexConstraint = cms.bool(False),
        beamSpot = cms.InputTag("hltOfflineBeamSpot")
    )
)