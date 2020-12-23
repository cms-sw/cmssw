import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.gpu_cff import gpu

# ECAL multifit running on CPU
from RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi import ecalMultiFitUncalibRecHit

ecalMultiFitUncalibRecHitTask = cms.Task(ecalMultiFitUncalibRecHit)

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
  digisLabelEB = cms.InputTag('ecalDigisGPU', 'ebDigis'),
  digisLabelEE = cms.InputTag('ecalDigisGPU', 'eeDigis'),
)

# copy the uncalibrated rechits from GPU to CPU
from RecoLocalCalo.EcalRecProducers.ecalCPUUncalibRecHitProducer_cfi import ecalCPUUncalibRecHitProducer as _ecalCPUUncalibRecHitProducer
ecalMultiFitUncalibRecHitSoA = _ecalCPUUncalibRecHitProducer.clone(
  recHitsInLabelEB = cms.InputTag('ecalMultiFitUncalibRecHitGPU', 'EcalUncalibRecHitsEB'),
  recHitsInLabelEE = cms.InputTag('ecalMultiFitUncalibRecHitGPU', 'EcalUncalibRecHitsEE'),
)

# convert the uncalibrated rechits from SoA to legacy format
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitConvertGPU2CPUFormat_cfi import ecalUncalibRecHitConvertGPU2CPUFormat as _ecalUncalibRecHitConvertGPU2CPUFormat
_ecalMultiFitUncalibRecHit_gpu = _ecalUncalibRecHitConvertGPU2CPUFormat.clone(
  recHitsLabelGPUEB = cms.InputTag('ecalMultiFitUncalibRecHitSoA', 'EcalUncalibRecHitsEB'),
  recHitsLabelGPUEE = cms.InputTag('ecalMultiFitUncalibRecHitSoA', 'EcalUncalibRecHitsEE'),
)
gpu.toReplaceWith(ecalMultiFitUncalibRecHit, _ecalMultiFitUncalibRecHit_gpu)

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
  # ECAL multifit running on GP
  ecalMultiFitUncalibRecHitGPU,
  # copy the uncalibrated rechits from GPU to CPU
  ecalMultiFitUncalibRecHitSoA,
  # convert the uncalibrated rechits legacy format
  ecalMultiFitUncalibRecHit,
))
