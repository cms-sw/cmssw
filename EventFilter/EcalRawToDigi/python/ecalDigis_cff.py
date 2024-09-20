import FWCore.ParameterSet.Config as cms

# ECAL unpacker running on CPU
from EventFilter.EcalRawToDigi.EcalUnpackerData_cfi import ecalEBunpacker as _ecalEBunpacker
ecalDigis = _ecalEBunpacker.clone()

ecalDigisTask = cms.Task(
    # ECAL unpacker running on CPU
    ecalDigis
)

from Configuration.StandardSequences.Accelerators_cff import *

# process modifier to run alpaka implementation
from Configuration.ProcessModifiers.alpaka_cff import alpaka

# ECAL conditions used by the portable unpacker
from EventFilter.EcalRawToDigi.ecalElectronicsMappingHostESProducer_cfi import ecalElectronicsMappingHostESProducer

# alpaka ECAL unpacker
from EventFilter.EcalRawToDigi.ecalRawToDigiPortable_cfi import ecalRawToDigiPortable as _ecalRawToDigiPortable
ecalDigisPortable = _ecalRawToDigiPortable.clone()

from EventFilter.EcalRawToDigi.ecalDigisFromPortableProducer_cfi import ecalDigisFromPortableProducer as _ecalDigisFromPortableProducer

# replace the SwitchProducer branches with a module to copy the ECAL digis from the accelerator to CPU (if needed) and convert them from SoA to legacy format
_ecalDigisFromPortable = _ecalDigisFromPortableProducer.clone(
    digisInLabelEB = 'ecalDigisPortable:ebDigis',
    digisInLabelEE = 'ecalDigisPortable:eeDigis',
    produceDummyIntegrityCollections = True
)
alpaka.toModify(ecalDigis,
    cpu = _ecalDigisFromPortable.clone()
)

alpaka.toReplaceWith(ecalDigisTask, cms.Task(
    # ECAL conditions used by the portable unpacker
    ecalElectronicsMappingHostESProducer,
    # run the portable ECAL unpacker
    ecalDigisPortable,
    # copy the ECAL digis from GPU to CPU (if needed) and convert them from SoA to legacy format
    ecalDigis
))
