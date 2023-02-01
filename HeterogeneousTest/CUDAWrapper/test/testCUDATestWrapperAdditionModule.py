import FWCore.ParameterSet.Config as cms

process = cms.Process('TestCUDATestWrapperAdditionModule')

process.source = cms.Source('EmptySource')

process.CUDAService = cms.Service('CUDAService')

process.cudaTestWrapperAdditionModule = cms.EDAnalyzer('CUDATestWrapperAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.cudaTestWrapperAdditionModule)

process.maxEvents.input = 1
