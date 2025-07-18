import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.prod = cms.EDProducer("PostInsertProducer")

process.path = cms.Path(
    process.prod
)
