import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitPhase2_cfi import ecalUncalibRecHitPhase2 as _ecalUncalibRecHitPhase2
ecalUncalibRecHitPhase2 = SwitchProducerCUDA(
  cpu = _ecalUncalibRecHitPhase2.clone()
)

# cpu weights
ecalUncalibRecHitPhase2Task = cms.Task(ecalUncalibRecHitPhase2)
