import FWCore.ParameterSet.Config as cms

process = cms.Process('TestAlpakaTestOpaqueAdditionModule')
process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

process.source = cms.Source('EmptySource')

process.alpakaTestDeviceAdditionModule = cms.EDAnalyzer('AlpakaTestDeviceAdditionModule@alpaka',
    size = cms.uint32( 1024*1024 )
)

process.alpakaTestKernelAdditionModule = cms.EDAnalyzer('AlpakaTestKernelAdditionModule@alpaka',
    size = cms.uint32( 1024*1024 )
)

process.alpakaTestWrapperAdditionModule = cms.EDAnalyzer('AlpakaTestWrapperAdditionModule@alpaka',
    size = cms.uint32( 1024*1024 )
)

process.alpakaTestOpaqueAdditionModule = cms.EDAnalyzer('AlpakaTestOpaqueAdditionModule@alpaka',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(
    # this one fails for the CUDA backend with "cudaErrorInvalidDeviceFunction: invalid device function"
    # process.alpakaTestDeviceAdditionModule +
    process.alpakaTestKernelAdditionModule +
    process.alpakaTestWrapperAdditionModule +
    process.alpakaTestOpaqueAdditionModule)

process.maxEvents.input = 1
