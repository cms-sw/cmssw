import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.OutOfTime_cff import *

StripCPEESProducer = cms.ESProducer("StripCPEESProducer",
                                    ComponentName = cms.string('SimpleStripCPE'),
                                    OutOfTime = OutOfTime
)


