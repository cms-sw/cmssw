import FWCore.ParameterSet.Config as cms

process = cms.Process("A")

process.maxEvents.input = 1
process.source = cms.Source("EmptySource")

process.load("FWCore.Services.DependencyGraph_cfi")
process.DependencyGraph.fileName = "test_module_delete_dependencygraph.gv"

intEventProducer = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
intEventProducerMustRun = cms.EDProducer("edmtest::MustRunIntProducer", ivalue = cms.int32(1), mustRunEvent = cms.bool(True))
intEventConsumer = cms.EDAnalyzer("IntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("producerEventConsumed"),
    valueMustMatch = cms.untracked.int32(1)
)
intGenericConsumer = cms.EDAnalyzer("edmtest::GenericIntsAnalyzer",
    srcEvent = cms.untracked.VInputTag(),
    inputShouldExist = cms.untracked.bool(True)
)

process.producerAEventConsumedInB = intEventProducer.clone(ivalue = 1)
process.producerAEventConsumedInBA = intEventProducer.clone(ivalue = 10)

process.producerEventNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteProducer")
process.producerBeginLumiNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteInLumiProducer")
process.producerBeginRunNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteInRunProducer")
process.producerBeginProcessNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteInProcessProducer")

# These producers do not get the event transitions for the events
# where the same-name producers in the SubProcesses produce a product.
# Nevertheless, these producers must not be deleted early, because
# their event transitions might get called.
process.producerEventMaybeConsumedInB = intEventProducerMustRun.clone(mustRunEvent=False)
process.producerEventMaybeConsumedInBA = intEventProducerMustRun.clone(mustRunEvent=False)

process.producerAEventNotConsumedInB = cms.EDProducer("edmtest::TestModuleDeleteProducer")
process.producerAEventNotConsumedInBA = cms.EDProducer("edmtest::TestModuleDeleteProducer")

process.producerEventConsumedInB1 = intEventProducerMustRun.clone()
process.producerEventConsumedInB2 = intEventProducerMustRun.clone()
process.producerEventConsumedInBA1 = intEventProducerMustRun.clone()
process.producerEventConsumedInBA2 = intEventProducerMustRun.clone()

process.intAnalyzerDelete = cms.EDAnalyzer("edmtest::TestModuleDeleteAnalyzer")

process.t = cms.Task(
    process.producerAEventConsumedInB,
    #
    process.producerAEventConsumedInBA,
    #
    process.producerEventNotConsumed,
    process.producerBeginLumiNotConsumed,
    process.producerBeginRunNotConsumed,
    process.producerBeginProcessNotConsumed,
    #
    process.producerEventMaybeConsumedInB,
    process.producerEventMaybeConsumedInBA,
    #
    process.producerAEventNotConsumedInB,
    process.producerAEventNotConsumedInBA,
    #
    process.producerEventConsumedInB1,
    process.producerEventConsumedInB2,
    process.producerEventConsumedInBA1,
    process.producerEventConsumedInBA2,
)

process.p = cms.Path(process.intAnalyzerDelete, process.t)

####################
subprocessB = cms.Process("B")
process.addSubProcess( cms.SubProcess(
    process = subprocessB,
    SelectEvents = cms.untracked.PSet(),
    outputCommands = cms.untracked.vstring()
) )

subprocessB.consumerEventFromA = intEventConsumer.clone(moduleLabel = "producerAEventConsumedInB", valueMustMatch = 1)

subprocessB.producerEventNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteProducer")

subprocessB.producerEventMaybeConsumedInB = intEventProducerMustRun.clone()
subprocessB.producerEventMaybeConsumedInBA = intEventProducerMustRun.clone(mustRunEvent=False)
subprocessB.consumerEventMaybeInB = intGenericConsumer.clone(srcEvent = ["producerEventMaybeConsumedInB"])

subprocessB.producerAEventNotConsumedInB = intEventProducerMustRun.clone()
subprocessB.producerAEventNotConsumedInBA = cms.EDProducer("edmtest::TestModuleDeleteProducer")
subprocessB.consumerAEventNotConsumedInB = intGenericConsumer.clone(srcEvent = ["producerAEventNotConsumedInB::B"])

subprocessB.producerEventConsumedInB1 = cms.EDProducer("edmtest::TestModuleDeleteProducer")
subprocessB.producerEventConsumedInB2 = cms.EDProducer("edmtest::TestModuleDeleteProducer")
subprocessB.consumerEventNotConsumedInB1 = intGenericConsumer.clone(srcEvent = ["producerEventConsumedInB1::A"])
subprocessB.consumerEventNotConsumedInB2 = intGenericConsumer.clone(srcEvent = [cms.InputTag("producerEventConsumedInB2", "", cms.InputTag.skipCurrentProcess())])
subprocessB.producerBEventConsumedInBA1 = intEventProducerMustRun.clone()
subprocessB.producerBEventConsumedInBA2 = intEventProducerMustRun.clone()

subprocessB.producerBEventConsumedInB1 = intEventProducerMustRun.clone()
subprocessB.producerBEventConsumedInB2 = intEventProducerMustRun.clone()
subprocessB.producerBEventConsumedInB3 = intEventProducerMustRun.clone()
subprocessB.consumerBEventConsumedInB1 = intGenericConsumer.clone(srcEvent = ["producerBEventConsumedInB1"])
subprocessB.consumerBEventConsumedInB2 = intGenericConsumer.clone(srcEvent = ["producerBEventConsumedInB2::B"])
subprocessB.consumerBEventConsumedInB3 = intGenericConsumer.clone(srcEvent = [cms.InputTag("producerBEventConsumedInB3", "", cms.InputTag.currentProcess())])


subprocessB.t = cms.Task(
    subprocessB.producerEventNotConsumed,
    #
    subprocessB.producerEventMaybeConsumedInB,
    subprocessB.producerEventMaybeConsumedInBA,
    #
    subprocessB.producerAEventNotConsumedInB,
    subprocessB.producerAEventNotConsumedInBA,
    #
    subprocessB.producerEventConsumedInB1,
    subprocessB.producerEventConsumedInB2,
    subprocessB.producerBEventConsumedInBA1,
    subprocessB.producerBEventConsumedInBA2,
    #
    subprocessB.producerBEventConsumedInB1,
    subprocessB.producerBEventConsumedInB2,
    subprocessB.producerBEventConsumedInB3,
)
subprocessB.p = cms.Path(
    subprocessB.consumerEventFromA+
    #
    subprocessB.consumerEventMaybeInB+
    #
    subprocessB.consumerAEventNotConsumedInB+
    subprocessB.consumerEventNotConsumedInB1+
    subprocessB.consumerEventNotConsumedInB2+
    #
    subprocessB.consumerBEventConsumedInB1+
    subprocessB.consumerBEventConsumedInB2+
    subprocessB.consumerBEventConsumedInB3
    ,subprocessB.t
)

####################
subprocessBA = cms.Process("BA")
subprocessB.addSubProcess( cms.SubProcess(
    process = subprocessBA,
    SelectEvents = cms.untracked.PSet(),
    outputCommands = cms.untracked.vstring()
) )

subprocessBA.consumerEventFromA = intEventConsumer.clone(moduleLabel = "producerAEventConsumedInBA", valueMustMatch = 10)

subprocessBA.producerEventMaybeConsumedInBA = intEventProducerMustRun.clone()
subprocessBA.consumerEventMaybeInBA = intGenericConsumer.clone(srcEvent = ["producerEventMaybeConsumedInBA"])

subprocessBA.producerAEventNotConsumedInBA = intEventProducerMustRun.clone()
subprocessBA.consumerAEventNotConsumedInBA = intGenericConsumer.clone(srcEvent = ["producerAEventNotConsumedInBA::BA"])

subprocessBA.producerEventConsumedInBA1 = cms.EDProducer("edmtest::TestModuleDeleteProducer")
subprocessBA.producerEventConsumedInBA2 = cms.EDProducer("edmtest::TestModuleDeleteProducer")
subprocessBA.producerBEventConsumedInBA1 = cms.EDProducer("edmtest::TestModuleDeleteProducer")
subprocessBA.producerBEventConsumedInBA2 = cms.EDProducer("edmtest::TestModuleDeleteProducer")
subprocessBA.consumerEventNotConsumedInBA1 = intGenericConsumer.clone(srcEvent = ["producerEventConsumedInBA1::A",
                                                                                  "producerBEventConsumedInBA1::B"])
subprocessBA.consumerEventNotConsumedInBA2 = intGenericConsumer.clone(srcEvent = [
    cms.InputTag("producerEventConsumedInBA2", "", cms.InputTag.skipCurrentProcess()),
    cms.InputTag("producerBEventConsumedInBA2", "", cms.InputTag.skipCurrentProcess())
])

subprocessBA.t = cms.Task(
    subprocessBA.producerEventMaybeConsumedInBA,
    #
    subprocessBA.producerAEventNotConsumedInBA,
    #
    subprocessBA.producerEventConsumedInBA1,
    subprocessBA.producerEventConsumedInBA2,
    subprocessBA.producerBEventConsumedInBA1,
    subprocessBA.producerBEventConsumedInBA2,
)
subprocessBA.p = cms.Path(
    subprocessBA.consumerEventFromA+
    #
    subprocessBA.consumerEventMaybeInBA+
    #
    subprocessBA.consumerAEventNotConsumedInBA+
    #
    subprocessBA.consumerEventNotConsumedInBA1+
    subprocessBA.consumerEventNotConsumedInBA2
    , subprocessBA.t
)
