import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.Phase2TrackerRecHits.Phase2StripCPEESProducer_cfi import *

phase2StripCPEGeometricESProducer = phase2StripCPEESProducer.clone()
#phase2StripCPEGeometricESProducer.ComponentName = cms.string('Phase2StripCPEGeometric')
phase2StripCPEGeometricESProducer.ComponentType = cms.string('Phase2StripCPEGeometric')
phase2StripCPEGeometricESProducer.parameters    = cms.PSet()
