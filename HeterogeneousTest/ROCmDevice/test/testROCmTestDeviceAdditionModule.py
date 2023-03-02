import FWCore.ParameterSet.Config as cms

process = cms.Process('TestROCmTestDeviceAdditionModule')
process.load('HeterogeneousCore.ROCmCore.ProcessAcceleratorROCm_cfi')

process.source = cms.Source('EmptySource')

process.rocmTestDeviceAdditionModule = cms.EDAnalyzer('ROCmTestDeviceAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.rocmTestDeviceAdditionModule)

process.maxEvents.input = 1
