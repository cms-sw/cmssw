import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.add_(cms.Service("ResourceEnforcer", maxTime = cms.untracked.double(0.01/60./60.)))

process.thing = cms.EDProducer("ThingProducer")

process.p = cms.Path(process.thing)

