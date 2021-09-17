import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.Phase2TrackerRecHits.Phase2StripCPEESProducer_cfi import *

phase2StripCPEGeometricESProducer = phase2StripCPEESProducer.clone(
    ComponentType = 'Phase2StripCPEGeometric',
    parameters    = cms.PSet()
)
