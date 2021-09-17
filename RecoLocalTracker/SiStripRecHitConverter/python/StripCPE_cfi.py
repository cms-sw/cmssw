import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPEESProducer_cfi import *
StripCPEESProducer = stripCPEESProducer.clone(
     ComponentName = 'SimpleStripCPE',
     ComponentType = 'SimpleStripCPE',
     parameters    = cms.PSet()
)
