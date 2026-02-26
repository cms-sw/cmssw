import FWCore.ParameterSet.Config as cms

# Legacy ECAL unpacker
from EventFilter.EcalRawToDigi.EcalUnpackerData_cfi import ecalEBunpacker as _ecalEBunpacker
ecalDigis = _ecalEBunpacker.clone()
ecalDigisLegacy = ecalDigis.clone()

ecalDigisTask = cms.Task(
    # Legacy ECAL unpacker
    ecalDigis
)

# remove unpacker until a Phase 2 version exists
from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
phase2_ecal_devel.toReplaceWith(ecalDigisTask, ecalDigisTask.copyAndExclude([ecalDigis]))

from Configuration.StandardSequences.Accelerators_cff import *

# process modifier to run alpaka implementation
from Configuration.ProcessModifiers.alpaka_cff import alpaka

# ECAL conditions used by the portable unpacker
from EventFilter.EcalRawToDigi.ecalElectronicsMappingHostESProducer_cfi import ecalElectronicsMappingHostESProducer
# Always enclose in a Task to prevent the construction of the
# ESProducer in the default configuration
ecalDigisPortableConditions = cms.Task(ecalElectronicsMappingHostESProducer)

# alpaka ECAL unpacker
from EventFilter.EcalRawToDigi.ecalRawToDigiPortable_cfi import ecalRawToDigiPortable as _ecalRawToDigiPortable
ecalDigisPortable = _ecalRawToDigiPortable.clone()

from EventFilter.EcalRawToDigi.ecalDigisFromPortableProducer_cfi import ecalDigisFromPortableProducer as _ecalDigisFromPortableProducer

# a module to convert them from SoA to legacy format
_ecalDigisFromPortable = _ecalDigisFromPortableProducer.clone(
    digisInLabelEB = 'ecalDigisPortable:ebDigis',
    digisInLabelEE = 'ecalDigisPortable:eeDigis',
    produceDummyIntegrityCollections = True
)
alpaka.toReplaceWith(ecalDigis, _ecalDigisFromPortable.clone())

ecalDigisTask_alpaka = cms.Task(
    # ECAL conditions used by the portable unpacker
    ecalDigisPortableConditions,
    # run the portable ECAL unpacker
    ecalDigisPortable,
    # convert them from SoA to legacy format
    ecalDigis
)

# remove portable unpacker until a Phase 2 version exists
phase2_ecal_devel.toReplaceWith(ecalDigisTask_alpaka, ecalDigisTask_alpaka.copyAndExclude([ecalDigisPortableConditions, ecalDigisPortable, ecalDigis]))

alpaka.toReplaceWith(ecalDigisTask, ecalDigisTask_alpaka)

# for GPU validation compare the legacy CPU module with the alpaka module
from Configuration.ProcessModifiers.gpuValidationEcal_cff import gpuValidationEcal
_ecalDigisTaskValidation = ecalDigisTask_alpaka.copy()
_ecalDigisTaskValidation.add(ecalDigisLegacy)
gpuValidationEcal.toReplaceWith(ecalDigisTask, _ecalDigisTaskValidation)

# for alpaka validation compare alpaka serial with alpaka
from Configuration.ProcessModifiers.alpakaValidationEcal_cff import alpakaValidationEcal
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone
ecalDigisPortableSerialSync = makeSerialClone(ecalDigisPortable)
ecalDigisSerialSync = _ecalDigisFromPortable.clone(
    digisInLabelEB = 'ecalDigisPortableSerialSync:ebDigis',
    digisInLabelEE = 'ecalDigisPortableSerialSync:eeDigis'
)
_ecalDigisTaskValidation = ecalDigisTask_alpaka.copy()
_ecalDigisTaskValidation.add(ecalDigisPortableSerialSync)
_ecalDigisTaskValidation.add(ecalDigisSerialSync)
alpakaValidationEcal.toReplaceWith(ecalDigisTask, _ecalDigisTaskValidation)
