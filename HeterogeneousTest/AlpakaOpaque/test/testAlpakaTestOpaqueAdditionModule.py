import FWCore.ParameterSet.Config as cms

process = cms.Process('TestAlpakaTestOpaqueAdditionModule')
process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

process.source = cms.Source('EmptySource')

process.alpakaTestOpaqueAdditionModule = cms.EDAnalyzer('AlpakaTestOpaqueAdditionModule@alpaka',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.alpakaTestOpaqueAdditionModule)

process.maxEvents.input = 1
