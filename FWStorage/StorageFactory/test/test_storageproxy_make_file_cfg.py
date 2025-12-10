import FWCore.ParameterSet.Config as cms

process = cms.Process("MAKE")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.out = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("test.root"))

process.Thing = cms.EDProducer("ThingProducer")
process.OtherThing = cms.EDProducer("OtherThingProducer")
process.EventNumber = cms.EDProducer("EventNumberIntProducer")


process.o = cms.EndPath(process.out, cms.Task(process.Thing, process.OtherThing, process.EventNumber))

