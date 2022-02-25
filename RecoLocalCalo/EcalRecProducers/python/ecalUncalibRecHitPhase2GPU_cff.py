import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
from Configuration.ProcessModifiers.gpu_cff import gpu


from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitPhase2_cfi import ecalUncalibRecHitPhase2 as _ecalUncalibRecHitPhase2
# ecalUncalibRecHitPhase2GPUTask = cms.Task(ecalUncalibRecHitPhase2GPU)
ecalUncalibRecHitPhase2new = SwitchProducerCUDA(
  cpu = _ecalUncalibRecHitPhase2.clone()
)

# cpu weights
ecalUncalibRecHitPhase2Task = cms.Task(ecalUncalibRecHitPhase2new)

# conditions used on gpu


from RecoLocalCalo.EcalRecProducers.ecalPh2DigiToGPUProducer_cfi import ecalPh2DigiToGPUProducer as _ecalPh2DigiToGPUProducer
ecalPh2DigiToGPUProducer = _ecalPh2DigiToGPUProducer.clone()

# gpu weights
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitPhase2GPU_cfi import ecalUncalibRecHitPhase2GPU as _ecalUncalibRecHitPhase2GPU
ecalUncalibRecHitPhase2GPU = _ecalUncalibRecHitPhase2GPU.clone(
  digisLabelEB = cms.InputTag('ecalPh2DigiToGPUProducer', 'ebDigis')
)

# copy the uncalibrated rechits from GPU to CPU
from RecoLocalCalo.EcalRecProducers.ecalCPUUncalibRecHitProducer_cfi import ecalCPUUncalibRecHitProducer as _ecalCPUUncalibRecHitProducer
ecalMultiFitUncalibRecHitSoAnew = _ecalCPUUncalibRecHitProducer.clone(
  recHitsInLabelEB = cms.InputTag('ecalUncalibRecHitPhase2GPU', 'EcalUncalibRecHitsEB')
  #recHitsInLabelEE = cms.InputTag('ecalUncalibRecHitPhase2GPU', 'EcalUncalibRecHitsEB'),
)


# convert the uncalibrated rechits from SoA to legacy format
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitConvertGPU2CPUFormat_cfi import ecalUncalibRecHitConvertGPU2CPUFormat as _ecalUncalibRecHitConvertGPU2CPUFormat
gpu.toModify(ecalUncalibRecHitPhase2new,
  cuda = _ecalUncalibRecHitConvertGPU2CPUFormat.clone(
  recHitsLabelGPUEB = cms.InputTag('ecalMultiFitUncalibRecHitSoAnew', 'EcalUncalibRecHitsEB')
# recHitsLabelGPUEE = cms.InputTag('ecalMultiFitUncalibRecHitSoAnew', 'EcalUncalibRecHitsEE')
    )
)

gpu.toReplaceWith(ecalUncalibRecHitPhase2Task, cms.Task(
  # ECAL conditions used by the multifit running on GPU
  # ecalPedestalsGPUESProducer,
  # ecalGainRatiosGPUESProducer,
  # ecalPulseShapesGPUESProducer,
  # ecalPulseCovariancesGPUESProducer,
  # ecalSamplesCorrelationGPUESProducer,
  # ecalTimeBiasCorrectionsGPUESProducer,
  # ecalTimeCalibConstantsGPUESProducer,
  # ecalMultifitParametersGPUESProducer,
  ecalPh2DigiToGPUProducer, 
  # ECAL weights running on GPU
  ecalUncalibRecHitPhase2GPU,
  # copy the uncalibrated rechits from GPU to CPU
  ecalMultiFitUncalibRecHitSoAnew,
  # ECAL multifit running on CPU, or convert the uncalibrated rechits from SoA to legacy format
  ecalUncalibRecHitPhase2new,
))
