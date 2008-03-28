import FWCore.ParameterSet.Config as cms

SiStripLAFakeESSource = cms.ESSource("SiStripLAFakeESSource",
    TemperatureError = cms.double(10.0),
    Temperature = cms.double(297.0),
    HoleRHAllParameter = cms.double(0.7),
    ChargeMobility = cms.double(480.0),
    HoleBeta = cms.double(1.213),
    HoleSaturationVelocity = cms.double(8370000.0),
    file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
    AppliedVoltage = cms.double(150.0)
)


