import FWCore.ParameterSet.Config as cms

process = cms.Process('TestCUDATestDeviceAdditionModule')

process.source = cms.Source('EmptySource')

process.CUDAService = cms.Service('CUDAService')

process.cudaTestDeviceAdditionModule = cms.EDAnalyzer('CUDATestDeviceAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.cudaTestDeviceAdditionModule)

process.maxEvents.input = 1
