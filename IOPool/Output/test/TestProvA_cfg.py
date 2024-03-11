import FWCore.ParameterSet.Config as cms
process = cms.Process("A")
process.source = cms.Source("EmptySource")
process.thing = cms.EDProducer("ThingProducer")
process.other = cms.EDProducer("OtherThingProducer", thingTag=cms.InputTag("thing"))
process.p = cms.Path(process.thing*process.other)
process.out = cms.OutputModule("PoolOutputModule", fileName=cms.untracked.string("a.root"))
process.test = cms.OutputModule("ProvenanceCheckerOutputModule")
process.o = cms.EndPath(process.test+process.out)
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(10))
# foo bar baz
