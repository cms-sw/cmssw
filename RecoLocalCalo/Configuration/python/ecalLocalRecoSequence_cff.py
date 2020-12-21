import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.gpu_cff import gpu

# TPG condition needed by ecalRecHit producer if TT recovery is ON
from RecoLocalCalo.EcalRecProducers.ecalRecHitTPGConditions_cff import *

# ECAL reconstruction
from RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cff import *
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalCompactTrigPrim_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalTPSkim_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalDetailedTimeRecHit_cfi import *

ecalUncalibRecHitTask = cms.Task(
    ecalMultiFitUncalibRecHitTask,
    ecalDetIdToBeRecovered)

ecalUncalibRecHitSequence = cms.Sequence(ecalUncalibRecHitTask)

ecalRecHitNoTPTask = cms.Task(
    ecalRecHit,
    ecalPreshowerRecHit)

ecalRecHitNoTPSequence = cms.Sequence(ecalRecHitNoTPTask)

ecalRecHitTask = cms.Task(
    ecalCompactTrigPrim,
    ecalTPSkim,
    ecalRecHitNoTPTask)

ecalRecHitSequence = cms.Sequence(ecalRecHitTask)

ecalLocalRecoTask = cms.Task(
    ecalUncalibRecHitTask,
    ecalRecHitTask)

ecalLocalRecoSequence = cms.Sequence(ecalLocalRecoTask)

ecalOnlyLocalRecoTask = cms.Task(
    ecalUncalibRecHitTask,
    ecalRecHitNoTPTask)

ecalOnlyLocalRecoSequence = cms.Sequence(ecalOnlyLocalRecoTask)

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

# convert the rechits from SoA to legacy format
from RecoLocalCalo.EcalRecProducers.ecalRecHitConvertGPU2CPUFormat_cfi import ecalRecHitConvertGPU2CPUFormat as _ecalRecHitConvertGPU2CPUFormat
_ecalRecHit_gpu = _ecalRecHitConvertGPU2CPUFormat.clone(
    recHitsLabelGPUEB = cms.InputTag('ecalRecHitSoA', 'EcalRecHitsEB'),
    recHitsLabelGPUEE = cms.InputTag('ecalRecHitSoA', 'EcalRecHitsEE')
)
# TODO: the ECAL calibrated rechits produced on the GPU are not correct, yet.
# When they are working and validated, remove this comment and uncomment the next line:
#gpu.toReplaceWith(ecalRecHit, _ecalRecHit_gpu)

# ECAL reconstruction on GPU
gpu.toReplaceWith(ecalRecHitNoTPTask, cms.Task(
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
  ecalRecHit,
  # ECAL preshower rechit legacy module
  ecalPreshowerRecHit
))

# Phase 2 modifications
from RecoLocalCalo.EcalRecProducers.ecalDetailedTimeRecHit_cfi import *
_phase2_timing_ecalRecHitTask = cms.Task( ecalRecHitTask.copy() , ecalDetailedTimeRecHit )
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toReplaceWith( ecalRecHitTask, _phase2_timing_ecalRecHitTask )

# FastSim modifications
_fastSim_ecalRecHitTask = ecalRecHitTask.copyAndExclude([ecalCompactTrigPrim,ecalTPSkim])
_fastSim_ecalUncalibRecHitTask = ecalUncalibRecHitTask.copyAndExclude([ecalDetIdToBeRecovered])
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(ecalRecHitTask, _fastSim_ecalRecHitTask)
fastSim.toReplaceWith(ecalUncalibRecHitTask, _fastSim_ecalUncalibRecHitTask)
