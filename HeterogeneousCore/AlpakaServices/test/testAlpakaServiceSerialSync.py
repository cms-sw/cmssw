import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32( 4 ),
    numberOfStreams = cms.untracked.uint32( 0 ),
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.AlpakaService = {}

from HeterogeneousCore.AlpakaServices.AlpakaServiceSerialSync_cfi import AlpakaServiceSerialSync as _AlpakaServiceSerialSync
process.AlpakaServiceSerialSync = _AlpakaServiceSerialSync.clone(
    verbose = True,
    hostAllocator = cms.untracked.PSet(
      binGrowth = cms.untracked.uint32(2),
      minBin = cms.untracked.uint32(8),
      maxBin = cms.untracked.uint32(30),
      maxCachedBytes = cms.untracked.uint64(0),
      maxCachedFraction = cms.untracked.double(0.8),
      fillAllocations = cms.untracked.bool(True),
      fillAllocationValue = cms.untracked.uint32(165),
      fillReallocations = cms.untracked.bool(True),
      fillReallocationValue = cms.untracked.uint32(90),
      fillDeallocations = cms.untracked.bool(True),
      fillDeallocationValue = cms.untracked.uint32(105),
      fillCaches = cms.untracked.bool(True),
      fillCacheValue = cms.untracked.uint32(150)
    ),
    deviceAllocator = cms.untracked.PSet(
      binGrowth = cms.untracked.uint32(2),
      minBin = cms.untracked.uint32(8),
      maxBin = cms.untracked.uint32(30),
      maxCachedBytes = cms.untracked.uint64(0),
      maxCachedFraction = cms.untracked.double(0.8),
      fillAllocations = cms.untracked.bool(True),
      fillAllocationValue = cms.untracked.uint32(165),
      fillReallocations = cms.untracked.bool(True),
      fillReallocationValue = cms.untracked.uint32(90),
      fillDeallocations = cms.untracked.bool(True),
      fillDeallocationValue = cms.untracked.uint32(105),
      fillCaches = cms.untracked.bool(True),
      fillCacheValue = cms.untracked.uint32(150)
    )
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 0 )
)
