import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.add_(cms.Service("ResourceEnforcer", maxVSize = cms.untracked.double(0.001)))

process.thing = cms.EDProducer("ThingProducer")

process.p = cms.Path(process.thing)

