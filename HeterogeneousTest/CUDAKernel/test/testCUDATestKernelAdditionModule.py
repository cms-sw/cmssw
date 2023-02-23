import FWCore.ParameterSet.Config as cms

process = cms.Process('TestCUDATestKernelAdditionModule')
process.load('HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA_cfi')

process.source = cms.Source('EmptySource')

process.cudaTestKernelAdditionModule = cms.EDAnalyzer('CUDATestKernelAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.cudaTestKernelAdditionModule)

process.maxEvents.input = 1
