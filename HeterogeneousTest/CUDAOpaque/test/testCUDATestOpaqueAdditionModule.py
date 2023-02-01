import FWCore.ParameterSet.Config as cms

process = cms.Process('TestCUDATestOpaqueAdditionModule')

process.source = cms.Source('EmptySource')

process.CUDAService = cms.Service('CUDAService')

process.cudaTestOpaqueAdditionModule = cms.EDAnalyzer('CUDATestOpaqueAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.cudaTestOpaqueAdditionModule)

process.maxEvents.input = 1
