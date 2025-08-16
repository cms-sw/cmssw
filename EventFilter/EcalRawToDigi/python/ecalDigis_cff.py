import FWCore.ParameterSet.Config as cms

# Legacy ECAL unpacker
from EventFilter.EcalRawToDigi.EcalUnpackerData_cfi import ecalEBunpacker as _ecalEBunpacker
ecalDigis = _ecalEBunpacker.clone()
ecalDigisLegacy = ecalDigis.clone()

ecalDigisTask = cms.Task(
    # Legacy ECAL unpacker
    ecalDigis
)

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

alpaka.toReplaceWith(ecalDigisTask, cms.Task(
    # ECAL conditions used by the portable unpacker
    ecalDigisPortableConditions,
    # run the portable ECAL unpacker
    ecalDigisPortable,
    # convert them from SoA to legacy format
    ecalDigis
))

# for alpaka validation compare the legacy CPU module with the alpaka module
from Configuration.ProcessModifiers.alpakaValidationEcal_cff import alpakaValidationEcal
_ecalDigisTaskValidation = ecalDigisTask.copy()
_ecalDigisTaskValidation.add(ecalDigisLegacy)
alpakaValidationEcal.toReplaceWith(ecalDigisTask, _ecalDigisTaskValidation)
