import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.add_(cms.Service("CPU"))

process.thing = cms.EDProducer("ThingProducer")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.p = cms.Path(process.thing)
# foo bar baz
# V8ZIzinP4xaP0
# K5j43E3BO2BUQ
