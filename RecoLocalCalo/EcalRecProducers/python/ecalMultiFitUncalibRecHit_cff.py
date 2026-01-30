import FWCore.ParameterSet.Config as cms

# Legacy ECAL multifit
from RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi import ecalMultiFitUncalibRecHit as _ecalMultiFitUncalibRecHit
ecalMultiFitUncalibRecHit = _ecalMultiFitUncalibRecHit.clone()
ecalMultiFitUncalibRecHitLegacy = ecalMultiFitUncalibRecHit.clone()

ecalMultiFitUncalibRecHitTask = cms.Task(
  # Legacy ECAL multifit
  ecalMultiFitUncalibRecHit
)

# modifications for alpaka
from Configuration.ProcessModifiers.alpaka_cff import alpaka

# ECAL conditions used by the multifit running on the accelerator
from RecoLocalCalo.EcalRecProducers.ecalMultifitConditionsHostESProducer_cfi import ecalMultifitConditionsHostESProducer
# Always enclose in a Task to prevent the construction of the
# ESProducer in the default configuration
ecalMultiFitUncalibRecHitPortableConditions = cms.Task(ecalMultifitConditionsHostESProducer)

# ECAL multifit running on the accelerator
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitProducerPortable_cfi import ecalUncalibRecHitProducerPortable as _ecalUncalibRecHitProducerPortable
ecalMultiFitUncalibRecHitPortable = _ecalUncalibRecHitProducerPortable.clone(
  digisLabelEB = 'ecalDigisPortable:ebDigis',
  digisLabelEE = 'ecalDigisPortable:eeDigis'
)

# a module to convert the uncalibrated rechits from SoA to legacy format
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitSoAToLegacy_cfi import ecalUncalibRecHitSoAToLegacy as _ecalUncalibRecHitSoAToLegacy
alpaka.toReplaceWith(ecalMultiFitUncalibRecHit, _ecalUncalibRecHitSoAToLegacy.clone())

alpaka.toReplaceWith(ecalMultiFitUncalibRecHitTask, cms.Task(
  # ECAL conditions used by the multifit running on the accelerator
  ecalMultiFitUncalibRecHitPortableConditions,
  # ECAL multifit running on device
  ecalMultiFitUncalibRecHitPortable,
  # convert the uncalibrated rechits from SoA to legacy format
  ecalMultiFitUncalibRecHit,
))

# for GPU validation compare the legacy CPU module with the alpaka module
from Configuration.ProcessModifiers.gpuValidationEcal_cff import gpuValidationEcal
_ecalMultiFitUncalibRecHitTaskValidation = ecalMultiFitUncalibRecHitTask.copy()
_ecalMultiFitUncalibRecHitTaskValidation.add(ecalMultiFitUncalibRecHitLegacy)
gpuValidationEcal.toReplaceWith(ecalMultiFitUncalibRecHitTask, _ecalMultiFitUncalibRecHitTaskValidation)

# for alpaka validation compare alpaka serial with alpaka
from Configuration.ProcessModifiers.alpakaValidationEcal_cff import alpakaValidationEcal
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone
ecalMultiFitUncalibRecHitPortableSerialSync = makeSerialClone(ecalMultiFitUncalibRecHitPortable)
ecalMultiFitUncalibRecHitSerialSync = _ecalUncalibRecHitSoAToLegacy.clone(
    inputCollectionEB = 'ecalMultiFitUncalibRecHitPortableSerialSync:EcalUncalibRecHitsEB',
    inputCollectionEE = 'ecalMultiFitUncalibRecHitPortableSerialSync:EcalUncalibRecHitsEE'
)
_ecalMultiFitUncalibRecHitTaskValidation = ecalMultiFitUncalibRecHitTask.copy()
_ecalMultiFitUncalibRecHitTaskValidation.add(ecalMultiFitUncalibRecHitPortableSerialSync)
_ecalMultiFitUncalibRecHitTaskValidation.add(ecalMultiFitUncalibRecHitSerialSync)
alpakaValidationEcal.toReplaceWith(ecalMultiFitUncalibRecHitTask, _ecalMultiFitUncalibRecHitTaskValidation)
