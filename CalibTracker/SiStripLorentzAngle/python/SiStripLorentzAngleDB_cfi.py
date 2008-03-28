# The following comments couldn't be translated into the new config version:

# input from trajectories

# calibration

import FWCore.ParameterSet.Config as cms

sistripLorentzAngle = cms.EDFilter("SiStripLorentzAngleDB",
    Temperature = cms.double(297.0),
    ChargeMobility = cms.double(480.0),
    HoleBeta = cms.double(1.213),
    HoleSaturationVelocity = cms.double(8370000.0),
    TOBXMax = cms.double(0.4),
    # th. calculation
    AppliedVoltage = cms.double(150.0),
    TemperatureError = cms.double(10.0),
    HoleRHAllParameter = cms.double(0.7),
    ModuleXMin = cms.double(-0.3),
    TIBFitXMin = cms.double(-0.3),
    Tracks = cms.InputTag("generalTracks"),
    TIBFitXMax = cms.double(0.3),
    ModuleXMax = cms.double(0.3),
    ModuleFitXMin = cms.double(-0.3),
    fileName = cms.string('trackhisto.root'),
    DoCalibration = cms.bool(False),
    #output files
    fitName = cms.string('fit_results'),
    ModuleFitXMax = cms.double(0.3),
    TOBFitXMin = cms.double(-0.3),
    TOBXMin = cms.double(-0.4),
    TOBFitXMax = cms.double(0.3),
    TIBXMin = cms.double(-0.3),
    TIBXMax = cms.double(0.3)
)


