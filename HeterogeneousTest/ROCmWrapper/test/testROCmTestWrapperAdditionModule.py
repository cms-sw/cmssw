import FWCore.ParameterSet.Config as cms

process = cms.Process('TestROCmTestWrapperAdditionModule')

process.source = cms.Source('EmptySource')

process.ROCmService = cms.Service('ROCmService')

process.rocmTestWrapperAdditionModule = cms.EDAnalyzer('ROCmTestWrapperAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.rocmTestWrapperAdditionModule)

process.maxEvents.input = 1
