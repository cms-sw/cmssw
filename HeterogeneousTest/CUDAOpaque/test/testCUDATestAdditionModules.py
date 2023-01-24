import FWCore.ParameterSet.Config as cms

process = cms.Process('TestCUDATestOpaqueAdditionModule')

process.source = cms.Source('EmptySource')

process.CUDAService = cms.Service('CUDAService')

process.cudaTestDeviceAdditionModule = cms.EDAnalyzer('CUDATestDeviceAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.cudaTestKernelAdditionModule = cms.EDAnalyzer('CUDATestKernelAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.cudaTestWrapperAdditionModule = cms.EDAnalyzer('CUDATestWrapperAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.cudaTestOpaqueAdditionModule = cms.EDAnalyzer('CUDATestOpaqueAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(
    # this one fails with "cudaErrorInvalidDeviceFunction: invalid device function"
    #process.cudaTestDeviceAdditionModule +
    process.cudaTestKernelAdditionModule +
    process.cudaTestWrapperAdditionModule +
    process.cudaTestOpaqueAdditionModule)

process.maxEvents.input = 1
