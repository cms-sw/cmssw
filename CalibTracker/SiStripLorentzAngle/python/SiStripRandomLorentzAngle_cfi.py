import FWCore.ParameterSet.Config as cms

sistripLorentzAngle = cms.EDFilter("SiStripRandomLorentzAngle",
    #parameters of the base class (ConditionDBWriter)
    IOVMode = cms.string('Run'),
    TemperatureError = cms.double(10.0),
    Temperature = cms.double(297.0),
    SinceAppendMode = cms.bool(True),
    HoleRHAllParameter = cms.double(0.7),
    ChargeMobility = cms.double(480.0),
    HoleBeta = cms.double(1.213),
    Record = cms.string('SiStripLorentzAngleRcd'),
    doStoreOnDB = cms.bool(True),
    HoleSaturationVelocity = cms.double(8370000.0),
    #parameters of the derived class
    AppliedVoltage = cms.double(150.0)
)


