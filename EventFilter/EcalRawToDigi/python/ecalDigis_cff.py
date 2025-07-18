import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

# ECAL unpacker running on CPU
from EventFilter.EcalRawToDigi.EcalUnpackerData_cfi import ecalEBunpacker as _ecalEBunpacker
ecalDigisCPU = _ecalEBunpacker.clone()

ecalDigis = SwitchProducerCUDA(
    cpu = ecalDigisCPU
)

ecalDigisTask = cms.Task(
    # ECAL unpacker running on CPU
    ecalDigis
)

from Configuration.StandardSequences.Accelerators_cff import *

# process modifier to run on GPUs
from Configuration.ProcessModifiers.gpu_cff import gpu

# ECAL conditions used by the unpacker running on GPU
from EventFilter.EcalRawToDigi.ecalElectronicsMappingGPUESProducer_cfi import ecalElectronicsMappingGPUESProducer

# ECAL unpacker running on GPU
from EventFilter.EcalRawToDigi.ecalRawToDigiGPU_cfi import ecalRawToDigiGPU as _ecalRawToDigiGPU
ecalDigisGPU = _ecalRawToDigiGPU.clone()

# extend the SwitchProducer to add a case to copy the ECAL digis from GPU to CPU and convert them from SoA to legacy format
from EventFilter.EcalRawToDigi.ecalCPUDigisProducer_cfi import ecalCPUDigisProducer as _ecalCPUDigisProducer
gpu.toModify(ecalDigis,
    # copy the ECAL digis from GPU to CPU and convert them from SoA to legacy format
    cuda = _ecalCPUDigisProducer.clone(
        digisInLabelEB = ('ecalDigisGPU', 'ebDigis'),
        digisInLabelEE = ('ecalDigisGPU', 'eeDigis'),
        produceDummyIntegrityCollections = True
    )
)

gpu.toReplaceWith(ecalDigisTask, cms.Task(
    # ECAL conditions used by the unpacker running on GPU
    ecalElectronicsMappingGPUESProducer,
    # run the ECAL unpacker on GPU
    ecalDigisGPU,
    # run the ECAL unpacker on CPU, or copy the ECAL digis from GPU to CPU and convert them from SoA to legacy format
    ecalDigis
))

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

# for alpaka validation compare the legacy CPU module with the alpaka module
from Configuration.ProcessModifiers.alpakaValidationEcal_cff import alpakaValidationEcal
alpakaValidationEcal.toModify(ecalDigis, cpu = ecalDigisCPU)
alpakaValidationEcal.toModify(ecalDigis, cuda = _ecalDigisFromPortable.clone())

