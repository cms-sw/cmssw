import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.add_(cms.Service("ResourceEnforcer",
                         maxVSize = cms.untracked.double(1.0),
                         maxRSS = cms.untracked.double(1.0),
                         maxTime = cms.untracked.double(1.0)))

process.thing = cms.EDProducer("ThingProducer")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.p = cms.Path(process.thing)

