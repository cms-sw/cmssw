import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.OutOfTime_cff import *

StripCPEfromTrackAngle2ESProducer = cms.ESProducer("StripCPEESProducer",
                                                   ComponentName = cms.string('StripCPEfromTrackAngle2'),
                                                   Temperature = cms.double(297.0),
                                                   HoleRHAllParameter = cms.double(0.7),
                                                   ChargeMobility = cms.double(480.0),
                                                   HoleBeta = cms.double(1.213),
                                                   HoleSaturationVelocity = cms.double(8370000.0),
                                                   AppliedVoltage = cms.double(150.0),
                                                   UseCalibrationFromDB = cms.bool(False),
                                                   OutOfTime = OutOfTime
)
