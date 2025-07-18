import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

process.source = cms.Source('EmptySource')

process.maxEvents.input = 3

process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

process.producer = cms.EDProducer('TestAlpakaGlobalProducerWithPtr@alpaka',
    size = cms.int32(32)
)
process.analyzer = cms.EDAnalyzer('TestAlpakaAnalyzerProductWithPtr',
    src = cms.InputTag('producer')
)

process.p = cms.Path(
    process.producer
    + process.analyzer
)
