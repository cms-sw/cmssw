import FWCore.ParameterSet.Config as cms

def customiseAlpakaServiceMemoryFilling(process):
  # load all variants of the AlpakaService
  # ProcessAcceleratorAlpaka will take care of removing the unused ones

  process.load('HeterogeneousCore.AlpakaServices.AlpakaServiceSerialSync_cfi')

  # load the CUDAService and the AlpakaService for the CUDA backend, if available
  try:
      process.load('HeterogeneousCore.CUDAServices.CUDAService_cfi')
      process.load('HeterogeneousCore.AlpakaServices.AlpakaServiceCudaAsync_cfi')
  except:
      pass

  # load the ROCmService and the AlpakaService for the ROCm backend, if available
  try:
      process.load('HeterogeneousCore.ROCmServices.ROCmService_cfi')
      process.load('HeterogeneousCore.AlpakaServices.AlpakaServiceROCmAsync_cfi')
  except:
      pass

  # enable junk memory filling for all AlpakaServices
  for name in process.services_():
    if name.startswith('AlpakaService'):
      service = getattr(process, name)
      # host allocator
      service.hostAllocator.fillAllocations = True
      service.hostAllocator.fillAllocationValue = 0xB4
      service.hostAllocator.fillReallocations = True
      service.hostAllocator.fillReallocationValue = 0x78
      service.hostAllocator.fillDeallocations = True
      service.hostAllocator.fillDeallocationValue = 0x4B
      service.hostAllocator.fillCaches = True
      service.hostAllocator.fillCacheValue = 0x87
      # device allocator
      service.deviceAllocator.fillAllocations = True
      service.deviceAllocator.fillAllocationValue = 0xA5
      service.deviceAllocator.fillReallocations = True
      service.deviceAllocator.fillReallocationValue = 0x69
      service.deviceAllocator.fillDeallocations = True
      service.deviceAllocator.fillDeallocationValue = 0x5A
      service.deviceAllocator.fillCaches = True
      service.deviceAllocator.fillCacheValue = 0x96

  return process
