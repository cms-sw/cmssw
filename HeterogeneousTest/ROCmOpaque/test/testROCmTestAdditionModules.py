import FWCore.ParameterSet.Config as cms

process = cms.Process('TestROCmTestOpaqueAdditionModule')
process.load('HeterogeneousCore.ROCmCore.ProcessAcceleratorROCm_cfi')

process.source = cms.Source('EmptySource')

process.rocmTestDeviceAdditionModule = cms.EDAnalyzer('ROCmTestDeviceAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.rocmTestKernelAdditionModule = cms.EDAnalyzer('ROCmTestKernelAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.rocmTestWrapperAdditionModule = cms.EDAnalyzer('ROCmTestWrapperAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.rocmTestOpaqueAdditionModule = cms.EDAnalyzer('ROCmTestOpaqueAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(
    process.rocmTestDeviceAdditionModule +
    process.rocmTestKernelAdditionModule +
    process.rocmTestWrapperAdditionModule +
    process.rocmTestOpaqueAdditionModule)

process.maxEvents.input = 1
