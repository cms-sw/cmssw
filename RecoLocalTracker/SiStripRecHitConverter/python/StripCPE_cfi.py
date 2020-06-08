import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPEESProducer_cfi import *
StripCPEESProducer = stripCPEESProducer.clone()
StripCPEESProducer.ComponentName = 'SimpleStripCPE'
StripCPEESProducer.ComponentType = 'SimpleStripCPE'
StripCPEESProducer.parameters    = cms.PSet()


