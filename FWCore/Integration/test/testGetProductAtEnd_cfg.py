import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(2))

process.thing = cms.EDProducer("ThingProducer")

process.getThing = cms.EDAnalyzer("edmtest::ThingAnalyzer")

process.e = cms.EndPath(process.thing+process.getThing)
