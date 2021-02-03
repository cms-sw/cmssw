import FWCore.ParameterSet.Config as cms

MuonUpdatorAtVertexAnyDirection = cms.PSet(
    MuonUpdatorAtVertexParameters = cms.PSet(
        BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3),
        MaxChi2 = cms.double(1000000.0),
        Propagator = cms.string('SteppingHelixPropagatorAny')
    )
)