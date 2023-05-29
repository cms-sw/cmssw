import FWCore.ParameterSet.Config as cms

process = cms.Process('TestROCmTestWrapperAdditionModule')
process.load('HeterogeneousCore.ROCmCore.ProcessAcceleratorROCm_cfi')

process.source = cms.Source('EmptySource')

process.rocmTestWrapperAdditionModule = cms.EDAnalyzer('ROCmTestWrapperAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.rocmTestWrapperAdditionModule)

process.maxEvents.input = 1
