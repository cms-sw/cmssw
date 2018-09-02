import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

process.source = cms.Source("EmptySource")

process.load('HeterogeneousCore.MPICore.stringProducer_cfi')
process.stringProducer.message = 'Hello world'

process.load('HeterogeneousCore.MPICore.genericConsumer_cfi')
process.genericConsumer.source = 'stringProducer'

process.load('HeterogeneousCore.MPICore.stringConsumer_cfi')
process.stringConsumer.source = cms.InputTag('genericConsumer', 'stringProducer')

process.path = cms.Path(
    process.stringProducer +
    process.genericConsumer +
    process.stringConsumer)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 1 )
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False ),
    numberOfThreads = cms.untracked.uint32( 1 ),
    numberOfStreams = cms.untracked.uint32( 0 )
)

