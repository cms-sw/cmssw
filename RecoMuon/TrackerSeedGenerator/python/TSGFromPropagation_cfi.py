import FWCore.ParameterSet.Config as cms

SeedGeneratorParameters = cms.PSet(
    ErrorRescaling = cms.double(3.0),
    ComponentName = cms.string('TSGFromPropagation'),
    errorMatrixPset = cms.PSet(

    ),
    UpdateState = cms.bool(False),
    UseSecondMeasurements = cms.bool(False),
    MaxChi2 = cms.double(30.0),
    UseVertexState = cms.bool(True),
    Propagator = cms.string('SmartPropagatorAnyOpposite')
)

