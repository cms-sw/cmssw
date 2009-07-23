import FWCore.ParameterSet.Config as cms

StripCPEfromTrackAngle2ESProducer = cms.ESProducer("StripCPEfromTrackAngle2ESProducer",
    Temperature = cms.double(297.0),
    ComponentName = cms.string('StripCPEfromTrackAngle2'),
    HoleRHAllParameter = cms.double(0.7),
    ChargeMobility = cms.double(480.0),
    HoleBeta = cms.double(1.213),
    HoleSaturationVelocity = cms.double(8370000.0),
    AppliedVoltage = cms.double(150.0),
    UseCalibrationFromDB = cms.bool(False)
)


