import FWCore.ParameterSet.Config as cms

MuonUpdatorAtVertex = cms.PSet(
    MuonUpdatorAtVertexParameters = cms.PSet(
        MaxChi2 = cms.double(1000000.0),
        Propagator = cms.string('SteppingHelixPropagatorOpposite'),
        BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3)
    )
)
MuonUpdatorAtVertexAnyDirection = cms.PSet(
    MuonUpdatorAtVertexParameters = cms.PSet(
        MaxChi2 = cms.double(1000000.0),
        Propagator = cms.string('SteppingHelixPropagatorAny'),
        BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3)
    )
)


# foo bar baz
# qc9rurpOPlTNu
# GJX8Hq617E3Dk
