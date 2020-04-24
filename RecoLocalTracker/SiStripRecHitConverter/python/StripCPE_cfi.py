import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPEESProducer_cfi import *
StripCPEESProducer = stripCPEESProducer.clone()
StripCPEESProducer.ComponentName = cms.string('SimpleStripCPE')
StripCPEESProducer.ComponentType = cms.string('SimpleStripCPE')
StripCPEESProducer.parameters    = cms.PSet()


