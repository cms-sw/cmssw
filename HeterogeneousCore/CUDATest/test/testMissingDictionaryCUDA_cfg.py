import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.producer = cms.EDProducer("edmtest::MissingDictionaryCUDAProducer")

process.path = cms.Path(process.producer)
