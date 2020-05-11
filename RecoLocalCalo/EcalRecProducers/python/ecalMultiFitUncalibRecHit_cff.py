import FWCore.ParameterSet.Config as cms

# ECAL multifit running on CPU
from RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi import ecalMultiFitUncalibRecHit

ecalMultiFitUncalibRecHitTask = cms.Task(ecalMultiFitUncalibRecHit)
