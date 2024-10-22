import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("EmptySource")
process.options.numberOfStreams = 2

process.a = cms.EDProducer("edmtest::one::WatchLumiBlocksProducer", transitions = cms.int32(0))

process.p = cms.Path(process.a)
