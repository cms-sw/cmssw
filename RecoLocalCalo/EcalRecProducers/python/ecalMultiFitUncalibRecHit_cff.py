import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
from Configuration.ProcessModifiers.gpu_cff import gpu

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

# ECAL conditions used by the multifit running on GPU
from RecoLocalCalo.EcalRecProducers.ecalPedestalsGPUESProducer_cfi import ecalPedestalsGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalGainRatiosGPUESProducer_cfi import ecalGainRatiosGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalPulseShapesGPUESProducer_cfi import ecalPulseShapesGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalPulseCovariancesGPUESProducer_cfi import ecalPulseCovariancesGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalSamplesCorrelationGPUESProducer_cfi import ecalSamplesCorrelationGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalTimeBiasCorrectionsGPUESProducer_cfi import ecalTimeBiasCorrectionsGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalTimeCalibConstantsGPUESProducer_cfi import ecalTimeCalibConstantsGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalMultifitParametersGPUESProducer_cfi import ecalMultifitParametersGPUESProducer

# ECAL multifit running on GPU
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitProducerGPU_cfi import ecalUncalibRecHitProducerGPU as _ecalUncalibRecHitProducerGPU
ecalMultiFitUncalibRecHitGPU = _ecalUncalibRecHitProducerGPU.clone(
  digisLabelEB = 'ecalDigisGPU:ebDigis',
  digisLabelEE = 'ecalDigisGPU:eeDigis',
)

# copy the uncalibrated rechits from GPU to CPU
from RecoLocalCalo.EcalRecProducers.ecalCPUUncalibRecHitProducer_cfi import ecalCPUUncalibRecHitProducer as _ecalCPUUncalibRecHitProducer
ecalMultiFitUncalibRecHitSoA = _ecalCPUUncalibRecHitProducer.clone(
  recHitsInLabelEB = 'ecalMultiFitUncalibRecHitGPU:EcalUncalibRecHitsEB',
  recHitsInLabelEE = 'ecalMultiFitUncalibRecHitGPU:EcalUncalibRecHitsEE',
  containsTimingInformation = True
)

# convert the uncalibrated rechits from SoA to legacy format
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitConvertGPU2CPUFormat_cfi import ecalUncalibRecHitConvertGPU2CPUFormat as _ecalUncalibRecHitConvertGPU2CPUFormat
gpu.toModify(ecalMultiFitUncalibRecHit,
  cuda = _ecalUncalibRecHitConvertGPU2CPUFormat.clone(
    recHitsLabelGPUEB = 'ecalMultiFitUncalibRecHitSoA:EcalUncalibRecHitsEB',
    recHitsLabelGPUEE = 'ecalMultiFitUncalibRecHitSoA:EcalUncalibRecHitsEE',
  )
)

gpu.toReplaceWith(ecalMultiFitUncalibRecHitTask, cms.Task(
  # ECAL conditions used by the multifit running on GPU
  ecalPedestalsGPUESProducer,
  ecalGainRatiosGPUESProducer,
  ecalPulseShapesGPUESProducer,
  ecalPulseCovariancesGPUESProducer,
  ecalSamplesCorrelationGPUESProducer,
  ecalTimeBiasCorrectionsGPUESProducer,
  ecalTimeCalibConstantsGPUESProducer,
  ecalMultifitParametersGPUESProducer,
  # ECAL multifit running on GPU
  ecalMultiFitUncalibRecHitGPU,
  # copy the uncalibrated rechits from GPU to CPU
  ecalMultiFitUncalibRecHitSoA,
  # ECAL multifit running on CPU, or convert the uncalibrated rechits from SoA to legacy format
  ecalMultiFitUncalibRecHit,
))

# modifications for alpaka
from Configuration.ProcessModifiers.alpaka_cff import alpaka

# ECAL conditions used by the multifit running on the accelerator
from RecoLocalCalo.EcalRecProducers.ecalMultifitConditionsHostESProducer_cfi import ecalMultifitConditionsHostESProducer

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
  # ECAL multifit running on device
  ecalMultiFitUncalibRecHitPortable,
  # ECAL multifit running on CPU, or convert the uncalibrated rechits from SoA to legacy format
  ecalMultiFitUncalibRecHit,
))

# for alpaka validation compare the legacy CPU module with the alpaka module
from Configuration.ProcessModifiers.alpakaValidationEcal_cff import alpakaValidationEcal
alpakaValidationEcal.toModify(ecalMultiFitUncalibRecHit, cpu = ecalMultiFitUncalibRecHitCPU)
alpakaValidationEcal.toModify(ecalMultiFitUncalibRecHit, cuda = _ecalUncalibRecHitSoAToLegacy.clone())

