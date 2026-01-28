import FWCore.ParameterSet.Config as cms

# Legacy ECAL calibrated rechit reconstruction
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import ecalRecHit as _ecalRecHit
ecalRecHit = _ecalRecHit.clone()
ecalRecHitLegacy = ecalRecHit.clone()

ecalCalibratedRecHitTask = cms.Task(
    ecalRecHit
)

# modifications for alpaka
from Configuration.ProcessModifiers.alpaka_cff import alpaka

# ECAL conditions used by the rechit producer running on the accelerator
from RecoLocalCalo.EcalRecProducers.ecalRecHitConditionsESProducer_cfi import ecalRecHitConditionsESProducer
# Always enclose in a Task to prevent the construction of the
# ESProducer in the default configuration
ecalRecHitPortableConditions = cms.Task(ecalRecHitConditionsESProducer)

# ECAL rechit producer running on the accelerator
from RecoLocalCalo.EcalRecProducers.ecalRecHitProducerPortable_cfi import ecalRecHitProducerPortable as _ecalRecHitProducerPortable
ecalRecHitPortable = _ecalRecHitProducerPortable.clone(
  uncalibrecHitsInLabelEB = 'ecalMultiFitUncalibRecHitPortable:EcalUncalibRecHitsEB',
  uncalibrecHitsInLabelEE = 'ecalMultiFitUncalibRecHitPortable:EcalUncalibRecHitsEE'
)

# a module to convert the rechits from SoA to legacy format
from RecoLocalCalo.EcalRecProducers.ecalRecHitSoAToLegacy_cfi import ecalRecHitSoAToLegacy as _ecalRecHitSoAToLegacy
# TODO: the portably produced ECAL calibrated rechits are not correct yet.
# When they are working and validated, remove this comment and uncomment the next lines:
#alpaka.toReplaceWith(ecalRecHit, _ecalRecHitSoAToLegacy.clone())

alpaka.toReplaceWith(ecalCalibratedRecHitTask, cms.Task(
  # ECAL conditions and parameters used by the rechit producer running on the accelerator
  ecalRecHitPortableConditions,
  # ECAL rechit producer running on device
  ecalRecHitPortable,
  # convert the rechits from SoA to legacy format
  ecalRecHit,
))

# for gpu validation compare the legacy CPU module with the alpaka module
from Configuration.ProcessModifiers.gpuValidationEcal_cff import gpuValidationEcal
_ecalCalibratedRecHitTaskValidation = ecalCalibratedRecHitTask.copy()
_ecalCalibratedRecHitTaskValidation.add()
gpuValidationEcal.toReplaceWith(ecalCalibratedRecHitTask, _ecalCalibratedRecHitTaskValidation)

# for alpaka validation compare alpaka serial with alpaka
from Configuration.ProcessModifiers.alpakaValidationEcal_cff import alpakaValidationEcal
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone
ecalRecHitPortableSerialSync = makeSerialClone(ecalRecHitPortable)
ecalRecHitSerialSync = _ecalRecHitSoAToLegacy.clone(
    inputCollectionEB = 'ecalRecHitPortableSerialSync:EcalRecHitsEB',
    inputCollectionEE = 'ecalRecHitPortableSerialSync:EcalRecHitsEE',
)
_ecalCalibratedRecHitTaskValidation = ecalCalibratedRecHitTask.copy()
_ecalCalibratedRecHitTaskValidation.add(ecalRecHitPortableSerialSync)
_ecalCalibratedRecHitTaskValidation.add(ecalRecHitSerialSync)
alpakaValidationEcal.toReplaceWith(ecalCalibratedRecHitTask, _ecalCalibratedRecHitTaskValidation)

from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
phase2_ecal_devel.toModify(ecalRecHitPortable, isPhase2 = True)
phase2_ecal_devel.toModify(ecalRecHitPortable, uncalibrecHitsInLabelEB = 'ecalUncalibRecHitPhase2SoA:EcalUncalibRecHitsEB')
phase2_ecal_devel.toModify(ecalRecHitPortable, EELaserMAX = None)
phase2_ecal_devel.toModify(ecalRecHitPortable, EELaserMIN = None)
phase2_ecal_devel.toModify(ecalRecHitPortable, recHitsLabelEE = None)
phase2_ecal_devel.toModify(ecalRecHitPortable, recoverEEFE = None)
phase2_ecal_devel.toModify(ecalRecHitPortable, recoverEEIsolatedChannels = None)
phase2_ecal_devel.toModify(ecalRecHitPortable, recoverEEVFE = None)
phase2_ecal_devel.toModify(ecalRecHitPortable, uncalibrecHitsInLabelEE = None)
