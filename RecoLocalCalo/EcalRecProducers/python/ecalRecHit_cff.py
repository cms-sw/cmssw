import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
from Configuration.ProcessModifiers.gpu_cff import gpu

# ECAL calibrated rechit reconstruction on CPU
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import ecalRecHit as _ecalRecHit
ecalRecHit = SwitchProducerCUDA(
    cpu = _ecalRecHit.clone()
)

ecalCalibratedRecHitTask = cms.Task(
    ecalRecHit
)

# ECAL rechit calibrations on GPU
from RecoLocalCalo.EcalRecProducers.ecalRechitADCToGeVConstantGPUESProducer_cfi import ecalRechitADCToGeVConstantGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalRechitChannelStatusGPUESProducer_cfi import ecalRechitChannelStatusGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalIntercalibConstantsGPUESProducer_cfi import ecalIntercalibConstantsGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalLaserAPDPNRatiosGPUESProducer_cfi import ecalLaserAPDPNRatiosGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalLaserAPDPNRatiosRefGPUESProducer_cfi import ecalLaserAPDPNRatiosRefGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalLaserAlphasGPUESProducer_cfi import ecalLaserAlphasGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalLinearCorrectionsGPUESProducer_cfi import ecalLinearCorrectionsGPUESProducer
from RecoLocalCalo.EcalRecProducers.ecalRecHitParametersGPUESProducer_cfi import ecalRecHitParametersGPUESProducer

# ECAL rechits running on GPU
from RecoLocalCalo.EcalRecProducers.ecalRecHitGPU_cfi import ecalRecHitGPU as _ecalRecHitGPU
ecalRecHitGPU = _ecalRecHitGPU.clone(
    uncalibrecHitsInLabelEB = cms.InputTag('ecalMultiFitUncalibRecHitGPU', 'EcalUncalibRecHitsEB'),
    uncalibrecHitsInLabelEE = cms.InputTag('ecalMultiFitUncalibRecHitGPU', 'EcalUncalibRecHitsEE')
)

# copy the rechits from GPU to CPU
from RecoLocalCalo.EcalRecProducers.ecalCPURecHitProducer_cfi import ecalCPURecHitProducer as _ecalCPURecHitProducer
ecalRecHitSoA = _ecalCPURecHitProducer.clone(
    recHitsInLabelEB = cms.InputTag('ecalRecHitGPU', 'EcalRecHitsEB'),
    recHitsInLabelEE = cms.InputTag('ecalRecHitGPU', 'EcalRecHitsEE')
)

# TODO: the ECAL calibrated rechits produced on the GPU are not correct, yet.
# When they are working and validated, remove this comment and uncomment the next lines:
# convert the rechits from SoA to legacy format
#from RecoLocalCalo.EcalRecProducers.ecalRecHitConvertGPU2CPUFormat_cfi import ecalRecHitConvertGPU2CPUFormat as _ecalRecHitFromSoA
#gpu.toModify(ecalRecHit,
#    cuda = _ecalRecHitFromSoA.clone(
#        recHitsLabelGPUEB = cms.InputTag('ecalRecHitSoA', 'EcalRecHitsEB'),
#        recHitsLabelGPUEE = cms.InputTag('ecalRecHitSoA', 'EcalRecHitsEE')
#    )
#)

# ECAL calibrated rechit reconstruction on GPU
gpu.toReplaceWith(ecalCalibratedRecHitTask, cms.Task(
  # ECAL rechit calibrations on GPU
  ecalRechitADCToGeVConstantGPUESProducer,
  ecalRechitChannelStatusGPUESProducer,
  ecalIntercalibConstantsGPUESProducer,
  ecalLaserAPDPNRatiosGPUESProducer,
  ecalLaserAPDPNRatiosRefGPUESProducer,
  ecalLaserAlphasGPUESProducer,
  ecalLinearCorrectionsGPUESProducer,
  ecalRecHitParametersGPUESProducer,
  # ECAL rechits running on GPU
  ecalRecHitGPU,
  # copy the rechits from GPU to CPU
  ecalRecHitSoA,
  # convert the rechits from SoA to legacy format
  ecalRecHit
))
