import FWCore.ParameterSet.Config as cms

process = cms.Process('TestAlpakaTestWrapperAdditionModule')
process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

process.source = cms.Source('EmptySource')

process.alpakaTestWrapperAdditionModule = cms.EDAnalyzer('AlpakaTestWrapperAdditionModule@alpaka',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.alpakaTestWrapperAdditionModule)

process.maxEvents.input = 1
