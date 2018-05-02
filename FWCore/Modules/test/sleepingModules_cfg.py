import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source =cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.options = cms.untracked.PSet(numberOfThreads = cms.untracked.uint32(4),
                                     numberOfStreams = cms.untracked.uint32(0))

#allows something like simulation of source
process.s1 = cms.EDProducer("timestudy::OneSleepingProducer",
    resource = cms.string("source"),
    ivalue = cms.int32(1),
    consumes = cms.VInputTag(),
    eventTimes = cms.vdouble(0.01,0.005))

process.s2 = cms.EDProducer("timestudy::OneSleepingProducer",
    resource = cms.string("source"),
    ivalue = cms.int32(2),
    consumes = cms.VInputTag(),
    eventTimes = cms.vdouble(0.02,0.03))

process.p1 = cms.EDProducer("timestudy::SleepingProducer",
    ivalue = cms.int32(3),
    consumes = cms.VInputTag("s1","s2"),
    eventTimes = cms.vdouble(0.05))


process.p2 = cms.EDProducer("timestudy::SleepingProducer",
    ivalue = cms.int32(3),
    consumes = cms.VInputTag("s2"),
    eventTimes = cms.vdouble(0.03))

process.p3 = cms.EDProducer("timestudy::SleepingProducer",
    ivalue = cms.int32(3),
    consumes = cms.VInputTag("p1","p2"),
    eventTimes = cms.vdouble(0.03))

#external work
process.add_(cms.Service("timestudy::SleepingServer",
                         nWaitingEvents = cms.untracked.uint32(4)))

process.e = cms.EDProducer("timestudy::ExternalWorkSleepingProducer",
                        consumes = cms.VInputTag("p2","p3"),
                        ivalue = cms.int32(10),
                        eventTimes = cms.vdouble(0.01),
                        serviceInitTimes = cms.vdouble(0.,0.),
                        serviceWorkTimes = cms.vdouble(0.1,0.15),
                        serviceFinishTimes = cms.vdouble(0.,0.)
)

#approximates an OutputModule
process.out = cms.EDAnalyzer("timestudy::OneSleepingAnalyzer",
    consumes = cms.VInputTag("s1","s2", "p1", "p2", "p3","e"),
    eventTimes = cms.vdouble(0.02,0.03)
)


process.o = cms.EndPath(process.out, cms.Task(process.s1,process.s2,process.p1,process.p2,process.p3,process.e))

#process.add_(cms.Service("Tracer"))

#process.add_(cms.Service("StallMonitor", fileName = cms.untracked.string("stall_sleep.log")))

process.add_(cms.Service("ZombieKillerService", secondsBetweenChecks = cms.untracked.uint32(10)))