import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.OutOfTime_cff import *

StripCPEfromTrackAngleESProducer = cms.ESProducer("StripCPEESProducer",
                                                  ComponentName = cms.string('StripCPEfromTrackAngle'),
                                                  OutOfTime = OutOfTime
)


