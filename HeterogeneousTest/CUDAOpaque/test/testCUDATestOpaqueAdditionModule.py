import FWCore.ParameterSet.Config as cms

process = cms.Process('TestCUDATestOpaqueAdditionModule')
process.load('HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA_cfi')

process.source = cms.Source('EmptySource')

process.cudaTestOpaqueAdditionModule = cms.EDAnalyzer('CUDATestOpaqueAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.cudaTestOpaqueAdditionModule)

process.maxEvents.input = 1
