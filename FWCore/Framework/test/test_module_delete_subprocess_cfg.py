import FWCore.ParameterSet.Config as cms

process = cms.Process("A")

process.maxEvents.input = 8
process.source = cms.Source("EmptySource")

#process.Tracer = cms.Service("Tracer")

# Process tree
# A (Process)
# ^
# \- B
# |  ^
# |  \- BA
# |
# \- C
# |
# \- D
#    ^
#    \- DA

# Cases to test
# - event/lumi/run/process product consumed in B or C, module kept
# - event/lumi/run/process product consumed in BA, module kept
# - event/lumi/run/process product not consumed anywhere, module deleted
# - event(/lumi/run) product produced in A and any SubProcess, consumed with empty process name, module kept
# - event(/lumi/run) product produced in A and any SubProcess, consumed with SubProcess name, A module deleted
# - event(/lumi/run) product produced in A and any SubProcess, consumed with A name or skipProcess, SubProcess module deleted
# - event(/lumi/run) product produced in B and consumed in BA, module kept
# - event(/lumi/run) product produced in B and consumed in C, module deleted (and product not found)
# - event(/lumi/run) product producer in A and dropped for SubProcess, module deleted

intEventProducer = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
intNonEventProducer = cms.EDProducer("NonEventIntProducer", ivalue = cms.int32(1))
intEventProducerMustRun = cms.EDProducer("edmtest::MustRunIntProducer", ivalue = cms.int32(1), mustRunEvent = cms.bool(True))
intEventConsumer = cms.EDAnalyzer("IntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("producerEventConsumed"),
    valueMustMatch = cms.untracked.int32(1)
)
intGenericConsumer = cms.EDAnalyzer("edmtest::GenericIntsAnalyzer",
    srcEvent = cms.untracked.VInputTag(),
    inputShouldExist = cms.untracked.bool(True)
)
uint64GenericConsumer = cms.EDAnalyzer("edmtest::GenericUInt64Analyzer",
    srcEvent = cms.untracked.VInputTag(),
    inputShouldExist = cms.untracked.bool(True)
)

def nonEventConsumer(transition, sourcePattern, expected):
    transCap = transition[0].upper() + transition[1:]
    blockName = transCap
    if "Lumi" in blockName:
        blockName = blockName+"nosityBlock"
    ret = intNonEventProducer.clone()
    setattr(ret, "consumes%s"%blockName, cms.InputTag(sourcePattern%transCap, transition))
    setattr(ret, "expect%s"%blockName, cms.untracked.int32(expected))
    return ret

process.producerAEventConsumedInB = intEventProducer.clone(ivalue = 1)
process.producerABeginLumiConsumedInB = intNonEventProducer.clone(ivalue = 2)
process.producerAEndRunConsumedInB = intNonEventProducer.clone(ivalue = 5)
process.producerABeginProcessBlockConsumedInB = intNonEventProducer.clone(ivalue = 6)
process.producerAEndLumiConsumedInC = intNonEventProducer.clone(ivalue = 3)
process.producerABeginRunConsumedInC = intNonEventProducer.clone(ivalue = 4)
process.producerAEndProcessBlockConsumedInC = intNonEventProducer.clone(ivalue = 7)

process.producerAEventConsumedInBA = intEventProducer.clone(ivalue = 10)
process.producerABeginLumiConsumedInBA = intNonEventProducer.clone(ivalue = 20)
process.producerAEndLumiConsumedInBA = intNonEventProducer.clone(ivalue = 30)
process.producerABeginRunConsumedInBA = intNonEventProducer.clone(ivalue = 40)
process.producerAEndRunConsumedInBA = intNonEventProducer.clone(ivalue = 50)
process.producerABeginProcessBlockConsumedInBA = intNonEventProducer.clone(ivalue = 60)
process.producerAEndProcessBlockConsumedInBA = intNonEventProducer.clone(ivalue = 70)

process.producerEventNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteProducer")
process.producerBeginLumiNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteInLumiProducer")
process.producerBeginRunNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteInRunProducer")
process.producerBeginProcessBlockNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteInProcessProducer")

# These producers do not get the event transitions for the events
# where the same-name producers in the SubProcesses produce a product.
# Nevertheless, these producers must not be deleted early, because
# their event transitions might get called.
process.producerEventMaybeConsumedInB = intEventProducerMustRun.clone(mustRunEvent=False)
process.producerEventMaybeConsumedInBA = intEventProducerMustRun.clone(mustRunEvent=False)
process.producerEventMaybeConsumedInC = intEventProducerMustRun.clone(mustRunEvent=False)

process.producerAEventNotConsumedInB = cms.EDProducer("edmtest::TestModuleDeleteProducer")
process.producerAEventNotConsumedInBA = cms.EDProducer("edmtest::TestModuleDeleteProducer")
process.producerAEventNotConsumedInC = cms.EDProducer("edmtest::TestModuleDeleteProducer")
process.producerAEventNotConsumedInD = cms.EDProducer("edmtest::TestModuleDeleteProducer")

process.producerEventConsumedInB1 = intEventProducerMustRun.clone()
process.producerEventConsumedInB2 = intEventProducerMustRun.clone()
process.producerEventConsumedInBA1 = intEventProducerMustRun.clone()
process.producerEventConsumedInBA2 = intEventProducerMustRun.clone()
process.producerEventConsumedInC1 = intEventProducerMustRun.clone()
process.producerEventConsumedInC2 = intEventProducerMustRun.clone()

process.producerANotConsumedChainEvent = cms.EDProducer("edmtest::TestModuleDeleteProducer")
process.producerANotConsumedChainLumi = cms.EDProducer("edmtest::TestModuleDeleteInLumiProducer",
    srcEvent = cms.untracked.VInputTag("producerANotConsumedChainEvent")
)
process.producerANotConsumedChainRun = cms.EDProducer("edmtest::TestModuleDeleteInRunProducer",
    srcEvent = cms.untracked.VInputTag("producerANotConsumedChainEvent")
)

process.intAnalyzerDelete = cms.EDAnalyzer("edmtest::TestModuleDeleteAnalyzer")

process.t = cms.Task(
    process.producerAEventConsumedInB,
    process.producerABeginLumiConsumedInB,
    process.producerAEndRunConsumedInB,
    process.producerABeginProcessBlockConsumedInB,
    process.producerAEndLumiConsumedInC,
    process.producerABeginRunConsumedInC,
    process.producerAEndProcessBlockConsumedInC,
    #
    process.producerAEventConsumedInBA,
    process.producerABeginLumiConsumedInBA,
    process.producerAEndLumiConsumedInBA,
    process.producerABeginRunConsumedInBA,
    process.producerAEndRunConsumedInBA,
    process.producerABeginProcessBlockConsumedInBA,
    process.producerAEndProcessBlockConsumedInBA,
    #
    process.producerEventNotConsumed,
    process.producerBeginLumiNotConsumed,
    process.producerBeginRunNotConsumed,
    process.producerBeginProcessBlockNotConsumed,
    #
    process.producerEventMaybeConsumedInB,
    process.producerEventMaybeConsumedInBA,
    process.producerEventMaybeConsumedInC,
    #
    process.producerAEventNotConsumedInB,
    process.producerAEventNotConsumedInBA,
    process.producerAEventNotConsumedInC,
    #
    process.producerEventConsumedInB1,
    process.producerEventConsumedInB2,
    process.producerEventConsumedInBA1,
    process.producerEventConsumedInBA2,
    process.producerEventConsumedInC1,
    process.producerEventConsumedInC2,
    #
    process.producerANotConsumedChainEvent,
    process.producerANotConsumedChainLumi,
    process.producerANotConsumedChainRun,
)


process.p = cms.Path(
    process.intAnalyzerDelete
    ,
    process.t
)

####################
subprocessB = cms.Process("B")
process.addSubProcess( cms.SubProcess(
    process = subprocessB,
    SelectEvents = cms.untracked.PSet(),
    outputCommands = cms.untracked.vstring()
) )

subprocessB.consumerEventFromA = intEventConsumer.clone(moduleLabel = "producerAEventConsumedInB", valueMustMatch = 1)
subprocessB.consumerBeginLumiFromA = nonEventConsumer("beginLumi", "producerA%sConsumedInB", 2)
subprocessB.consumerEndRunFromA = nonEventConsumer("endRun", "producerA%sConsumedInB", 5)
subprocessB.consumerBeginProcessBlockFromA = nonEventConsumer("beginProcessBlock", "producerA%sConsumedInB", 6)

subprocessB.consumerAEventNotConsumed = intGenericConsumer.clone(
    srcEvent = [
        "producerEventNotConsumed:doesNotExist",
        "producerEventNotConsumed:doesNotExist:A",
        cms.InputTag("producerEventNotConsumed", "doesNotExist", cms.InputTag.skipCurrentProcess())
    ],
    inputShouldExist = False,
)
subprocessB.consumerAEventNotConsumed2 = uint64GenericConsumer.clone(
    srcEvent = [
        "producerEventNotConsumed",
        "producerEventNotConsumed::A",
        cms.InputTag("producerEventNotConsumed", "", cms.InputTag.skipCurrentProcess())
    ],
    inputShouldExist = False,
)
subprocessB.producerEventNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteProducer")
subprocessB.producerBeginLumiNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteInLumiProducer")
subprocessB.producerBeginRunNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteInRunProducer")
subprocessB.producerBeginProcessBlockNotConsumed = cms.EDProducer("edmtest::TestModuleDeleteInProcessProducer")

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

subprocessB.producerBEventConsumedInC1 = cms.EDProducer("edmtest::TestModuleDeleteProducer")
subprocessB.producerBEventConsumedInC2 = cms.EDProducer("edmtest::TestModuleDeleteProducer")

subprocessB.producerBEventConsumedInB1 = intEventProducerMustRun.clone()
subprocessB.producerBEventConsumedInB2 = intEventProducerMustRun.clone()
subprocessB.producerBEventConsumedInB3 = intEventProducerMustRun.clone()
subprocessB.consumerBEventConsumedInB1 = intGenericConsumer.clone(srcEvent = ["producerBEventConsumedInB1"])
subprocessB.consumerBEventConsumedInB2 = intGenericConsumer.clone(srcEvent = ["producerBEventConsumedInB2::B"])
subprocessB.consumerBEventConsumedInB3 = intGenericConsumer.clone(srcEvent = [cms.InputTag("producerBEventConsumedInB3", "", cms.InputTag.currentProcess())])

subprocessB.producerBNotConsumedChainEvent = cms.EDProducer("edmtest::TestModuleDeleteProducer",
    srcBeginRun = cms.untracked.VInputTag("producerANotConsumedChainRun")
)
subprocessB.producerBNotConsumedChainLumi = cms.EDProducer("edmtest::TestModuleDeleteInLumiProducer",
    srcEvent = cms.untracked.VInputTag("producerANotConsumedChainEvent")
)
subprocessB.producerBNotConsumedChainRun = cms.EDProducer("edmtest::TestModuleDeleteInRunProducer",
    srcBeginLumi = cms.untracked.VInputTag("producerANotConsumedChainLumi")
)

subprocessB.t = cms.Task(
    subprocessB.producerEventNotConsumed,
    subprocessB.producerBeginLumiNotConsumed,
    subprocessB.producerBeginRunNotConsumed,
    subprocessB.producerBeginProcessBlockNotConsumed,
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
    subprocessB.producerBEventConsumedInC1,
    subprocessB.producerBEventConsumedInC2,
    #
    subprocessB.producerBEventConsumedInB1,
    subprocessB.producerBEventConsumedInB2,
    subprocessB.producerBEventConsumedInB3,
    #
    subprocessB.producerBNotConsumedChainEvent,
    subprocessB.producerBNotConsumedChainLumi,
    subprocessB.producerBNotConsumedChainRun,
)
subprocessB.p = cms.Path(
    subprocessB.consumerEventFromA+
    subprocessB.consumerBeginLumiFromA+
    subprocessB.consumerEndRunFromA+
    subprocessB.consumerBeginProcessBlockFromA+
    #
    subprocessB.consumerAEventNotConsumed+
    subprocessB.consumerAEventNotConsumed2+
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
subprocessBA.consumerBeginLumiFromA = nonEventConsumer("beginLumi", "producerA%sConsumedInBA", 20)
subprocessBA.consumerEndLumiFromA = nonEventConsumer("endLumi", "producerA%sConsumedInBA", 30)
subprocessBA.consumerBeginRunFromA = nonEventConsumer("beginRun", "producerA%sConsumedInBA", 40)
subprocessBA.consumerEndRunFromA = nonEventConsumer("endRun", "producerA%sConsumedInBA", 50)
subprocessBA.consumerBeginProcessBlockFromA = nonEventConsumer("beginProcessBlock", "producerA%sConsumedInBA", 60)
subprocessBA.consumerEndProcessBlockFromA = nonEventConsumer("endProcessBlock", "producerA%sConsumedInBA", 70)

subprocessBA.consumerABEventNotConsumed = intGenericConsumer.clone(
    srcEvent = [
        "producerEventNotConsumed:doesNotExist",
        "producerEventNotConsumed:doesNotExist:A",
        "producerEventNotConsumed:doesNotExist:B",
        cms.InputTag("producerEventNotConsumed", "doesNotExist", cms.InputTag.skipCurrentProcess())
    ],
    inputShouldExist = False,
)
subprocessBA.consumerABEventNotConsumed2 = uint64GenericConsumer.clone(
    srcEvent = [
        "producerEventNotConsumed",
        "producerEventNotConsumed::A",
        "producerEventNotConsumed::B",
        cms.InputTag("producerEventNotConsumed", "", cms.InputTag.skipCurrentProcess())
    ],
    inputShouldExist = False,
)

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

subprocessBA.producerBANotConsumedChainEvent = cms.EDProducer("edmtest::TestModuleDeleteProducer",
    srcBeginLumi = cms.untracked.VInputTag("producerBNotConsumedChainLumi")
)
subprocessBA.producerBANotConsumedChainLumi = cms.EDProducer("edmtest::TestModuleDeleteInLumiProducer",
    srcBeginRun = cms.untracked.VInputTag("producerBNotConsumedChainRun")
)
subprocessBA.producerBANotConsumedChainRun = cms.EDProducer("edmtest::TestModuleDeleteInRunProducer",
    srcEvent = cms.untracked.VInputTag("producerBNotConsumedChainEvent")
)
subprocessBA.producerBANotConsumedChainEvent2 = cms.EDProducer("edmtest::TestModuleDeleteProducer",
    srcBeginRun = cms.untracked.VInputTag("producerANotConsumedChainRun"),
    srcBeginLumi = cms.untracked.VInputTag("producerANotConsumedChainLumi"),
    srcEvent = cms.untracked.VInputTag("producerANotConsumedChainEvent"),
)


subprocessBA.t = cms.Task(
    subprocessBA.producerEventMaybeConsumedInBA,
    #
    subprocessBA.producerAEventNotConsumedInBA,
    #
    subprocessBA.producerEventConsumedInBA1,
    subprocessBA.producerEventConsumedInBA2,
    subprocessBA.producerBEventConsumedInBA1,
    subprocessBA.producerBEventConsumedInBA2,
    #
    subprocessBA.producerBANotConsumedChainEvent,
    subprocessBA.producerBANotConsumedChainLumi,
    subprocessBA.producerBANotConsumedChainRun,
    subprocessBA.producerBANotConsumedChainEvent2,
)
subprocessBA.p = cms.Path(
    subprocessBA.consumerEventFromA+
    subprocessBA.consumerBeginLumiFromA+
    subprocessBA.consumerEndLumiFromA+
    subprocessBA.consumerBeginRunFromA+
    subprocessBA.consumerEndRunFromA+
    subprocessBA.consumerBeginProcessBlockFromA+
    subprocessBA.consumerEndProcessBlockFromA+
    #
    subprocessBA.consumerABEventNotConsumed+
    subprocessBA.consumerABEventNotConsumed2+
    #
    subprocessBA.consumerEventMaybeInBA+
    #
    subprocessBA.consumerAEventNotConsumedInBA+
    #
    subprocessBA.consumerEventNotConsumedInBA1+
    subprocessBA.consumerEventNotConsumedInBA2
    , subprocessBA.t
)

####################
subprocessC = cms.Process("C")
process.addSubProcess( cms.SubProcess(
    process = subprocessC,
    SelectEvents = cms.untracked.PSet(),
    outputCommands = cms.untracked.vstring()
) )

subprocessC.consumerEndLumiFromA = nonEventConsumer("endLumi", "producerA%sConsumedInC", 3)
subprocessC.consumerBeginRunFromA = nonEventConsumer("beginRun", "producerA%sConsumedInC", 4)
subprocessC.consumerEndProcessBlockFromA = nonEventConsumer("endProcessBlock", "producerA%sConsumedInC", 7)

subprocessC.consumerAEventNotConsumed = intGenericConsumer.clone(
    srcEvent = [
        "producerEventNotConsumed:doesNotExist",
        "producerEventNotConsumed:doesNotExist:A",
        cms.InputTag("producerEventNotConsumed", "doesNotExist", cms.InputTag.skipCurrentProcess())
    ],
    inputShouldExist = False,
)
subprocessC.consumerAEventNotConsumed2 = uint64GenericConsumer.clone(
    srcEvent = [
        "producerEventNotConsumed",
        "producerEventNotConsumed::A",
        cms.InputTag("producerEventNotConsumed", "", cms.InputTag.skipCurrentProcess())
    ],
    inputShouldExist = False,
)

subprocessC.producerEventMaybeConsumedInC = intEventProducerMustRun.clone()
subprocessC.consumerEventMaybeInC = intGenericConsumer.clone(srcEvent = ["producerEventMaybeConsumedInC"])

subprocessC.producerAEventNotConsumedInC = intEventProducerMustRun.clone()
subprocessC.consumerAEventNotConsumedInC = intGenericConsumer.clone(srcEvent = ["producerAEventNotConsumedInC::C"])

subprocessC.producerEventConsumedInC1 = cms.EDProducer("edmtest::TestModuleDeleteProducer")
subprocessC.producerEventConsumedInC2 = cms.EDProducer("edmtest::TestModuleDeleteProducer")
subprocessC.consumerEventNotConsumedInC1 = intGenericConsumer.clone(srcEvent = ["producerEventConsumedInC1::A"])
subprocessC.consumerEventNotConsumedInC2 = intGenericConsumer.clone(srcEvent = [cms.InputTag("producerEventConsumedInC2", "", cms.InputTag.skipCurrentProcess())])

subprocessC.consumerBEventConsumedInC1 = intGenericConsumer.clone(srcEvent = ["producerBEventConsumedInC1"], inputShouldExist=False)
subprocessC.consumerBEventConsumedInC2 = intGenericConsumer.clone(srcEvent = ["producerBEventConsumedInC1::B"], inputShouldExist=False)

subprocessC.producerCNotConsumedChainEvent = cms.EDProducer("edmtest::TestModuleDeleteProducer",
    srcEvent = cms.untracked.VInputTag("producerANotConsumedChainEvent")
)
subprocessC.producerCNotConsumedChainLumi = cms.EDProducer("edmtest::TestModuleDeleteInLumiProducer",
    srcBeginLumi = cms.untracked.VInputTag("producerANotConsumedChainLumi")
)
subprocessC.producerCNotConsumedChainRun = cms.EDProducer("edmtest::TestModuleDeleteInRunProducer",
    srcBeginRun = cms.untracked.VInputTag("producerANotConsumedChainRun")
)

subprocessC.t = cms.Task(
    subprocessC.producerEventMaybeConsumedInC,
    #
    subprocessC.producerAEventNotConsumedInC,
    #
    subprocessC.producerEventConsumedInC1,
    subprocessC.producerEventConsumedInC2,
    #
    subprocessC.producerCNotConsumedChainEvent,
    subprocessC.producerCNotConsumedChainLumi,
    subprocessC.producerCNotConsumedChainRun,
)
subprocessC.p = cms.Path(
    subprocessC.consumerEndLumiFromA+
    subprocessC.consumerBeginRunFromA+
    subprocessC.consumerEndProcessBlockFromA+
    #
    subprocessC.consumerAEventNotConsumed+
    subprocessC.consumerAEventNotConsumed2+
    #
    subprocessC.consumerEventMaybeInC+
    subprocessC.consumerAEventNotConsumedInC+
    #
    subprocessC.consumerEventNotConsumedInC1+
    subprocessC.consumerEventNotConsumedInC2+
    #
    subprocessC.consumerBEventConsumedInC1+
    subprocessC.consumerBEventConsumedInC2
    , subprocessC.t
)

####################
subprocessD = cms.Process("D")
process.addSubProcess( cms.SubProcess(
    process = subprocessD,
    SelectEvents = cms.untracked.PSet(),
    outputCommands = cms.untracked.vstring(
        "drop *_producerAEventNotConsumedInD_*_*",
    )
) )

subprocessD.consumerAEventNotConsumedInD = intGenericConsumer.clone(
    srcEvent = [
        "producerAEvenNotConsumedInD",
        "producerAEvenNotConsumedInD::A",
        cms.InputTag("producerEventANotConsumedInD", "", cms.InputTag.skipCurrentProcess())
    ],
    inputShouldExist = False,
)

subprocessD.producerDEventNotConsumedInDA = cms.EDProducer("edmtest::TestModuleDeleteProducer")

subprocessD.t = cms.Task(
    subprocessD.producerDEventNotConsumedInDA
)
subprocessD.p = cms.Path(
    subprocessD.consumerAEventNotConsumedInD,
    subprocessD.t
)

####################
subprocessDA = cms.Process("BA")
subprocessD.addSubProcess( cms.SubProcess(
    process = subprocessDA,
    SelectEvents = cms.untracked.PSet(),
    outputCommands = cms.untracked.vstring(
        "drop *_producerDEventNotConsumedInDA_*_*",
    )
) )

subprocessDA.consumerAEventNotConsumedInD = intGenericConsumer.clone(
    srcEvent = [
        "producerAEvenNotConsumedInD",
        "producerAEvenNotConsumedInD::A",
        cms.InputTag("producerEventANotConsumedInD", "", cms.InputTag.skipCurrentProcess())
    ],
    inputShouldExist = False,
)
subprocessDA.consumerDEventNotConsumedInDA = intGenericConsumer.clone(
    srcEvent = [
        "producerDEvenNotConsumedInDA",
        "producerDEvenNotConsumedInDA::D",
        cms.InputTag("producerEventDNotConsumedInDA", "", cms.InputTag.skipCurrentProcess())
    ],
    inputShouldExist = False,
)

subprocessDA.producerDNotConsumedChainRun = cms.EDProducer("edmtest::TestModuleDeleteInRunProducer",
    srcBeginRun = cms.untracked.VInputTag("producerANotConsumedChainRun"),
    srcBeginLumi = cms.untracked.VInputTag("producerANotConsumedChainLumi"),
    srcEvent = cms.untracked.VInputTag("producerANotConsumedChainEvent"),
)

subprocessDA.t = cms.Task(
    subprocessDA.producerDNotConsumedChainRun
)

subprocessDA.p = cms.Path(
    subprocessDA.consumerAEventNotConsumedInD+
    subprocessDA.consumerDEventNotConsumedInDA,
    subprocessDA.t
)
