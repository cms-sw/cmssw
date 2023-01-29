import FWCore.ParameterSet.Config as cms

process = cms.Process('TestAlpakaTestDeviceAdditionModule')
process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

process.source = cms.Source('EmptySource')

process.alpakaTestDeviceAdditionModule = cms.EDAnalyzer('AlpakaTestDeviceAdditionModule@alpaka',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.alpakaTestDeviceAdditionModule)

process.maxEvents.input = 1
