import FWCore.ParameterSet.Config as cms

process = cms.Process('TestROCmTestDeviceAdditionModule')

process.source = cms.Source('EmptySource')

process.ROCmService = cms.Service('ROCmService')

process.rocmTestDeviceAdditionModule = cms.EDAnalyzer('ROCmTestDeviceAdditionModule',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.rocmTestDeviceAdditionModule)

process.maxEvents.input = 1
