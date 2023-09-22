import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

# ECAL unpacker running on CPU
from EventFilter.EcalRawToDigi.EcalUnpackerData_cfi import ecalEBunpacker as _ecalEBunpacker
ecalDigis = SwitchProducerCUDA(
    cpu = _ecalEBunpacker.clone()
)

ecalDigisTask = cms.Task(
    # ECAL unpacker running on CPU
    ecalDigis
)

# process modifier to run on GPUs
from Configuration.ProcessModifiers.gpu_cff import gpu

# ECAL conditions used by the unpacker running on GPU
from EventFilter.EcalRawToDigi.ecalElectronicsMappingGPUESProducer_cfi import ecalElectronicsMappingGPUESProducer

# ECAL unpacker running on GPU
from EventFilter.EcalRawToDigi.ecalRawToDigiGPU_cfi import ecalRawToDigiGPU as _ecalRawToDigiGPU
ecalDigisGPU = _ecalRawToDigiGPU.clone()

# extend the SwitchProducer to add a case to copy the ECAL digis from GPU to CPU and covert them from SoA to legacy format
from EventFilter.EcalRawToDigi.ecalCPUDigisProducer_cfi import ecalCPUDigisProducer as _ecalCPUDigisProducer
gpu.toModify(ecalDigis,
    # copy the ECAL digis from GPU to CPU and covert them from SoA to legacy format
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
    # run the ECAL unpacker on CPU, or copy the ECAL digis from GPU to CPU and covert them from SoA to legacy format
    ecalDigis
))
