import FWCore.ParameterSet.Config as cms

process = cms.Process('TestROCmTestKernelAdditionModule')

process.source = cms.Source('EmptySource')

process.ROCmService = cms.Service('ROCmService')

process.rocmTestKernelAdditionModule = cms.EDAnalyzer('ROCmTestKernelAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.rocmTestKernelAdditionModule)

process.maxEvents.input = 1
