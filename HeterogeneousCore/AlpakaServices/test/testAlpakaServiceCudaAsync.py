import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32( 4 ),
    numberOfStreams = cms.untracked.uint32( 0 ),
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.CUDAService = {}
process.MessageLogger.AlpakaService = {}

process.load('HeterogeneousCore.CUDAServices.CUDAService_cfi')

from HeterogeneousCore.AlpakaServices.AlpakaServiceCudaAsync_cfi import AlpakaServiceCudaAsync as _AlpakaServiceCudaAsync
process.AlpakaServiceCudaAsync = _AlpakaServiceCudaAsync.clone(
    verbose = True,
    hostAllocator = dict(
      binGrowth = 2,
      minBin = 8,                           # 256 bytes
      maxBin = 30,                          #   1 GB
      maxCachedBytes = 64*1024*1024*1024,   #  64 GB
      maxCachedFraction = 0.8,              # or 80%, whatever is less
      fillAllocations = True,
      fillAllocationValue = 0xA5,
      fillReallocations = True,
      fillReallocationValue = 0x69,
      fillDeallocations = True,
      fillDeallocationValue = 0x5A,
      fillCaches = True,
      fillCacheValue = 0x96
    ),
    deviceAllocator = dict(
      binGrowth = 2,
      minBin = 8,                           # 256 bytes
      maxBin = 30,                          #   1 GB
      maxCachedBytes = 8*1024*1024*1024,    #   8 GB
      maxCachedFraction = 0.8,              # or 80%, whatever is less
      fillAllocations = True,
      fillAllocationValue = 0xA5,
      fillReallocations = True,
      fillReallocationValue = 0x69,
      fillDeallocations = True,
      fillDeallocationValue = 0x5A,
      fillCaches = True,
      fillCacheValue = 0x96
    )
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 0 )
)
