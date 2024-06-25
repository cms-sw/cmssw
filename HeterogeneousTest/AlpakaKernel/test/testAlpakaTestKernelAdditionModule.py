import FWCore.ParameterSet.Config as cms

process = cms.Process('TestAlpakaTestKernelAdditionModule')
process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

process.source = cms.Source('EmptySource')

process.alpakaTestKernelAdditionModule = cms.EDAnalyzer('AlpakaTestKernelAdditionModule@alpaka',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.alpakaTestKernelAdditionModule)

process.maxEvents.input = 1
