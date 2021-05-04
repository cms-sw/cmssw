import FWCore.ParameterSet.Config as cms

# legacy raw to digi on the CPU
from EventFilter.EcalRawToDigi.EcalUnpackerData_cfi import ecalEBunpacker as _ecalEBunpacker
ecalDigis = _ecalEBunpacker.clone()

ecalDigisTask = cms.Task(ecalDigis)

# process modifier to run on GPUs
from Configuration.ProcessModifiers.gpu_cff import gpu

# GPU-friendly EventSetup modules
from EventFilter.EcalRawToDigi.ecalElectronicsMappingGPUESProducer_cfi import ecalElectronicsMappingGPUESProducer

# raw to digi on GPUs
from EventFilter.EcalRawToDigi.ecalRawToDigiGPU_cfi import ecalRawToDigiGPU as _ecalRawToDigiGPU
ecalDigisGPU = _ecalRawToDigiGPU.clone()

# copy the digi from the GPU to the CPU and convert to legacy format
from EventFilter.EcalRawToDigi.ecalCPUDigisProducer_cfi import ecalCPUDigisProducer as _ecalCPUDigisProducer
_ecalDigis_gpu = _ecalCPUDigisProducer.clone(
  digisInLabelEB = ('ecalDigisGPU', 'ebDigis'),
  digisInLabelEE = ('ecalDigisGPU', 'eeDigis'),
  produceDummyIntegrityCollections = True
)
gpu.toReplaceWith(ecalDigis, _ecalDigis_gpu)

gpu.toReplaceWith(ecalDigisTask, cms.Task(ecalElectronicsMappingGPUESProducer, ecalDigisGPU, ecalDigis))
