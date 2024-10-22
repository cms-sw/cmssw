import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource",
                            numberEventsInLuminosityBlock = cms.untracked.uint32(5))

process.thing = cms.EDProducer("ThingProducer", offsetDelta = cms.int32(1))


process.sleep = cms.EDProducer("timestudy::SleepingProducer",
    ivalue = cms.int32(1),
    consumes = cms.VInputTag(),
    eventTimes = cms.vdouble(1.0, 0.001, 0.001))

process.o = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("test_overlap_lumi.root"))

process.p = cms.Path( process.sleep )

process.maxEvents.input = 20

process.options.numberOfThreads = 8

process.e = cms.EndPath(process.o, cms.Task(process.thing))
