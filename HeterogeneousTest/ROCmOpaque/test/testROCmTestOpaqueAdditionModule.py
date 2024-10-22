import FWCore.ParameterSet.Config as cms

process = cms.Process('TestROCmTestOpaqueAdditionModule')
process.load('HeterogeneousCore.ROCmCore.ProcessAcceleratorROCm_cfi')

process.source = cms.Source('EmptySource')

process.rocmTestOpaqueAdditionModule = cms.EDAnalyzer('ROCmTestOpaqueAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.rocmTestOpaqueAdditionModule)

process.maxEvents.input = 1
