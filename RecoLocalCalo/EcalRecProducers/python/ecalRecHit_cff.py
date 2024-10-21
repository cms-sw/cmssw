import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
from Configuration.ProcessModifiers.gpu_cff import gpu

# ECAL calibrated rechit reconstruction on CPU
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import ecalRecHit as _ecalRecHit
ecalRecHitCPU = _ecalRecHit.clone()
ecalRecHit = SwitchProducerCUDA(
    cpu = ecalRecHitCPU
)

ecalCalibratedRecHitTask = cms.Task(
    ecalRecHit
)

from Configuration.StandardSequences.Accelerators_cff import *

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

# modifications for alpaka
from Configuration.ProcessModifiers.alpaka_cff import alpaka

# ECAL conditions used by the rechit producer running on the accelerator
from RecoLocalCalo.EcalRecProducers.ecalRecHitConditionsHostESProducer_cfi import ecalRecHitConditionsHostESProducer
from RecoLocalCalo.EcalRecProducers.ecalRecHitParametersHostESProducer_cfi import ecalRecHitParametersHostESProducer

ecalRecHitParametersSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalRecHitParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

# ECAL rechit producer running on the accelerator
from RecoLocalCalo.EcalRecProducers.ecalRecHitProducerPortable_cfi import ecalRecHitProducerPortable as _ecalRecHitProducerPortable
ecalRecHitPortable = _ecalRecHitProducerPortable.clone(
  uncalibrecHitsInLabelEB = 'ecalMultiFitUncalibRecHitPortable:EcalUncalibRecHitsEB',
  uncalibrecHitsInLabelEE = 'ecalMultiFitUncalibRecHitPortable:EcalUncalibRecHitsEE'
)

# replace the SwitchProducerCUDA branches with the module to convert the rechits from SoA to legacy format
from RecoLocalCalo.EcalRecProducers.ecalRecHitSoAToLegacy_cfi import ecalRecHitSoAToLegacy as _ecalRecHitSoAToLegacy
# TODO: the portably produced ECAL calibrated rechits correct yet.
# When they are working and validated, remove this comment and uncomment the next lines:
#alpaka.toModify(ecalRecHit,
#    cpu = _ecalRecHitSoAToLegacy.clone()
#)

alpaka.toReplaceWith(ecalCalibratedRecHitTask, cms.Task(
  # ECAL conditions and parameters used by the rechit producer running on the accelerator
  ecalRecHitConditionsHostESProducer,
  ecalRecHitParametersHostESProducer,
  # ECAL rechit producer running on device
  ecalRecHitPortable,
  # ECAL rechit producer running on CPU, or convert the rechits from SoA to legacy format
  ecalRecHit,
))

# for alpaka validation compare the legacy CPU module with the alpaka module
from Configuration.ProcessModifiers.alpakaValidationEcal_cff import alpakaValidationEcal
alpakaValidationEcal.toModify(ecalRecHit, cpu = ecalRecHitCPU)
alpakaValidationEcal.toModify(ecalRecHit, cuda = _ecalRecHitSoAToLegacy.clone())

