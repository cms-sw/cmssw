import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('HeterogeneousCore.CUDAServices.CUDAService_cfi')

# Empty source
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32( 4 ),
    numberOfStreams = cms.untracked.uint32( 4 ),
)

# Path and EndPath definitions
from HeterogeneousCore.Producer.testGPUConcurrency_cfi import testGPUConcurrency
process.testGPU = testGPUConcurrency.clone()
process.testGPU.sleep = 1000000
process.testGPU.blocks = 100000
process.testGPU.threads = 256

process.path = cms.Path(process.testGPU)

process.schedule = cms.Schedule(process.path)
