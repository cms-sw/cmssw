import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

# ECAL multifit running on CPU
from RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi import ecalMultiFitUncalibRecHit as _ecalMultiFitUncalibRecHit
ecalMultiFitUncalibRecHitCPU = _ecalMultiFitUncalibRecHit.clone()
ecalMultiFitUncalibRecHit = SwitchProducerCUDA(
  cpu = ecalMultiFitUncalibRecHitCPU
)

ecalMultiFitUncalibRecHitTask = cms.Task(
  # ECAL multifit running on CPU
  ecalMultiFitUncalibRecHit
)

from Configuration.StandardSequences.Accelerators_cff import *

# modifications for alpaka
from Configuration.ProcessModifiers.alpaka_cff import alpaka

# ECAL conditions used by the multifit running on the accelerator
from RecoLocalCalo.EcalRecProducers.ecalMultifitConditionsHostESProducer_cfi import ecalMultifitConditionsHostESProducer
from RecoLocalCalo.EcalRecProducers.ecalMultifitParametersHostESProducer_cfi import ecalMultifitParametersHostESProducer

ecalMultifitParametersSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalMultifitParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

# ECAL multifit running on the accelerator
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitProducerPortable_cfi import ecalUncalibRecHitProducerPortable as _ecalUncalibRecHitProducerPortable
ecalMultiFitUncalibRecHitPortable = _ecalUncalibRecHitProducerPortable.clone(
  digisLabelEB = 'ecalDigisPortable:ebDigis',
  digisLabelEE = 'ecalDigisPortable:eeDigis'
)

# replace the SwitchProducerCUDA branches with the module to convert the uncalibrated rechits from SoA to legacy format
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitSoAToLegacy_cfi import ecalUncalibRecHitSoAToLegacy as _ecalUncalibRecHitSoAToLegacy
alpaka.toModify(ecalMultiFitUncalibRecHit,
    cpu = _ecalUncalibRecHitSoAToLegacy.clone()
)

alpaka.toReplaceWith(ecalMultiFitUncalibRecHitTask, cms.Task(
  # ECAL conditions used by the multifit running on the accelerator
  ecalMultifitConditionsHostESProducer,
  ecalMultifitParametersHostESProducer,
  # ECAL multifit running on device
  ecalMultiFitUncalibRecHitPortable,
  # ECAL multifit running on CPU, or convert the uncalibrated rechits from SoA to legacy format
  ecalMultiFitUncalibRecHit,
))
