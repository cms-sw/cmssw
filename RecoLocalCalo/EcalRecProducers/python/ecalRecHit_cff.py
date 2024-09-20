import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

# ECAL calibrated rechit reconstruction on CPU
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import ecalRecHit as _ecalRecHit
ecalRecHit = SwitchProducerCUDA(
    cpu = _ecalRecHit.clone()
)

ecalCalibratedRecHitTask = cms.Task(
    ecalRecHit
)
