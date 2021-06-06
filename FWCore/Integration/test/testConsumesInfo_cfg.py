import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD1")

process.Tracer = cms.Service('Tracer',
    dumpPathsAndConsumes = cms.untracked.bool(True)
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        Tracer = cms.untracked.PSet(
            limit = cms.untracked.int32(100000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("IntSource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testConsumesInfo.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_intProducerA_*_*',
        'drop *_intProducerB_*_*',
        'drop *_intProducerC_*_*',
        'drop *_intProducerD_*_*',
        'drop *_intProducerE_*_*',
        'drop *_intProducerF_*_*',
        'drop *_intProducerG_*_*',
        'drop *_intProducerH_*_*',
        'drop *_intProducerI_*_*'
    )
)

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("source") ),
  expectedSum = cms.untracked.int32(12),
  inputTagsNotFound = cms.untracked.VInputTag(
    cms.InputTag("source", processName=cms.InputTag.skipCurrentProcess()),
    cms.InputTag("intProducer", processName=cms.InputTag.skipCurrentProcess()),
    cms.InputTag("intProducerU", processName=cms.InputTag.skipCurrentProcess())
  )
)

process.a2 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerA") ),
  expectedSum = cms.untracked.int32(300)
)

process.a3 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("aliasForInt") ),
  expectedSum = cms.untracked.int32(300)
)

process.a4 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerA", processName=cms.InputTag.currentProcess()) ),
  expectedSum = cms.untracked.int32(300)
)

process.intProducer = cms.EDProducer("IntProducer", ivalue = cms.int32(1))

process.intProducerU = cms.EDProducer("IntProducer", ivalue = cms.int32(10))

process.intProducerA = cms.EDProducer("IntProducer", ivalue = cms.int32(100))

process.aliasForInt = cms.EDAlias(
  intProducerA  = cms.VPSet(
    cms.PSet(type = cms.string('edmtestIntProduct')
    )
  )
)

process.intVectorProducer = cms.EDProducer("IntVectorProducer",
  count = cms.int32(9),
  ivalue = cms.int32(11)
)

process.test = cms.EDAnalyzer("TestContextAnalyzer",
                              pathname = cms.untracked.string("p"),
                              modlable = cms.untracked.string("test"))

process.testView1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsView = cms.untracked.VInputTag( cms.InputTag("intVectorProducer", "", "PROD1") )
)

process.testStreamingProducer = cms.EDProducer("IntProducer",
  ivalue = cms.int32(111)
)

process.testManyConsumingProducer = cms.EDProducer("ConsumingIntProducer",
                                               ivalue = cms.int32(111)
                                               )


process.testStreamingAnalyzer = cms.EDAnalyzer("ConsumingStreamAnalyzer",
  valueMustMatch = cms.untracked.int32(111),
  moduleLabel = cms.untracked.string("testStreamingProducer")
)

process.intProducerB = cms.EDProducer("IntProducer", ivalue = cms.int32(1000))
process.intProducerC = cms.EDProducer("IntProducer", ivalue = cms.int32(1001))
process.intProducerD = cms.EDProducer("IntProducer", ivalue = cms.int32(1002))
process.intProducerE = cms.EDProducer("IntProducer", ivalue = cms.int32(1003))
process.intProducerF = cms.EDProducer("IntProducer", ivalue = cms.int32(1004))
process.intProducerG = cms.EDProducer("IntProducer", ivalue = cms.int32(1005))
process.intProducerH = cms.EDProducer("IntProducer", ivalue = cms.int32(1006))
process.intProducerI = cms.EDProducer("IntProducer", ivalue = cms.int32(1007))

process.intProducerBeginProcessBlock = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(10000))

process.intProducerEndProcessBlock = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(100000))

process.processBlockTest1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  expectedSum = cms.untracked.int32(220000),
  inputTagsBeginProcessBlock = cms.untracked.VInputTag(
    cms.InputTag("intProducerBeginProcessBlock"),
    cms.InputTag("intProducerBeginProcessBlock", "", "PROD1")
  ),
  inputTagsEndProcessBlock = cms.untracked.VInputTag(
    cms.InputTag("intProducerEndProcessBlock"),
    cms.InputTag("intProducerEndProcessBlock", "", "PROD1"),
  )
)

process.p = cms.Path(process.intProducer * process.a1 * process.a2 * process.a3 *
                     process.a4 *
                     process.test * process.testView1 *
                     process.testStreamingProducer * process.testStreamingAnalyzer *
                     process.intProducerBeginProcessBlock *
                     process.intProducerEndProcessBlock *
                     process.processBlockTest1
)

process.p2 = cms.Path(process.intProducer * process.a1 * process.a2 * process.a3)

process.p3 = cms.Path(
    process.intProducerB *
    process.intProducerC *
    process.intProducerD *
    process.intProducerE
)


process.p11 = cms.Path()

process.t = cms.Task(
    process.intProducerU,
    process.intProducerA,
    process.intVectorProducer,
    process.intProducerF,
    process.intProducerG,
    process.intProducerH,
    process.intProducerI
)

process.e = cms.EndPath(process.testManyConsumingProducer+process.out, process.t)

process.p1ep2 = cms.EndPath()

process2 = cms.Process("PROD2")

process.addSubProcess(cms.SubProcess(process2,
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_intProducerA_*_*"
    )
))

process2.intProducerB = cms.EDProducer("IntProducer", ivalue = cms.int32(2000))
process2.intProducerD = cms.EDProducer("IntProducer", ivalue = cms.int32(2002))
process2.intProducerF = cms.EDProducer("IntProducer", ivalue = cms.int32(2004))
process2.intProducerH = cms.EDProducer("IntProducer", ivalue = cms.int32(2006))

process2.task1 = cms.Task(
    process2.intProducerF,
    process2.intProducerH
)

process2.path1 = cms.Path(
    process2.intProducerB *
    process2.intProducerD,
    process2.task1
)

copyProcess = cms.Process("COPY")
process2.addSubProcess(cms.SubProcess(copyProcess,
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_intProducerA_*_*"
    )
))

copyProcess.intVectorProducer = cms.EDProducer("IntVectorProducer",
  count = cms.int32(9),
  ivalue = cms.int32(12)
)

copyProcess.testView1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsView = cms.untracked.VInputTag( cms.InputTag("intVectorProducer", "", "PROD1") ),
  expectedSum = cms.untracked.int32(33)
)

copyProcess.testView2 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsView = cms.untracked.VInputTag( cms.InputTag("intVectorProducer", "", "COPY") ),
  expectedSum = cms.untracked.int32(36)
)

copyProcess.testView3 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsView = cms.untracked.VInputTag( cms.InputTag("intVectorProducer", "", processName=cms.InputTag.currentProcess()) ),
  expectedSum = cms.untracked.int32(36)
)

copyProcess.test = cms.EDAnalyzer("TestContextAnalyzer",
                                  pathname = cms.untracked.string("p3"),
                                  modlable = cms.untracked.string("test")
)

copyProcess.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")
copyProcess.testMergeResults = cms.EDAnalyzer("TestMergeResults")

copyProcess.testStreamingProducer = cms.EDProducer("IntProducer",
  ivalue = cms.int32(11)
)
copyProcess.testManyConsumingProducer = cms.EDProducer("ConsumingIntProducer",
                                                   ivalue = cms.int32(11)
                                                   )


copyProcess.testStreamingAnalyzer = cms.EDAnalyzer("ConsumingStreamAnalyzer",
  valueMustMatch = cms.untracked.int32(11),
  moduleLabel = cms.untracked.string("testStreamingProducer")
)

copyProcess.p3 = cms.Path(copyProcess.intVectorProducer * copyProcess.test * copyProcess.thingWithMergeProducer *
                          copyProcess.testMergeResults * copyProcess.testView1 * copyProcess.testView2 * copyProcess.testView3 *
                          copyProcess.testStreamingProducer * copyProcess.testStreamingAnalyzer)

copyProcess.ep1 = cms.EndPath(copyProcess.intVectorProducer+copyProcess.testManyConsumingProducer)
copyProcess.ep2 = cms.EndPath()

copyProcess.intProducerB = cms.EDProducer("IntProducer", ivalue = cms.int32(3000))
copyProcess.intProducerC = cms.EDProducer("IntProducer", ivalue = cms.int32(3001))
copyProcess.intProducerF = cms.EDProducer("IntProducer", ivalue = cms.int32(3004))
copyProcess.intProducerG = cms.EDProducer("IntProducer", ivalue = cms.int32(3005))

copyProcess.task1 = cms.Task(
    copyProcess.intProducerF,
    copyProcess.intProducerG
)

copyProcess.path1 = cms.Path(
    copyProcess.intProducerB *
    copyProcess.intProducerC,
    copyProcess.task1
)

copyProcess.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerB") ),
  expectedSum = cms.untracked.int32(9000)
)

copyProcess.a2 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerB", processName=cms.InputTag.skipCurrentProcess()) ),
  expectedSum = cms.untracked.int32(6000)
)

copyProcess.a3 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerC") ),
  expectedSum = cms.untracked.int32(9003)
)

copyProcess.a4 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerC", processName=cms.InputTag.skipCurrentProcess()) ),
  expectedSum = cms.untracked.int32(3003)
)

copyProcess.a5 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerD") ),
  expectedSum = cms.untracked.int32(6006)
)

copyProcess.a6 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerD", processName=cms.InputTag.skipCurrentProcess()) ),
  expectedSum = cms.untracked.int32(6006)
)

copyProcess.a7 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerE") ),
  expectedSum = cms.untracked.int32(3009)
)

copyProcess.a8 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerE", processName=cms.InputTag.skipCurrentProcess()) ),
  expectedSum = cms.untracked.int32(3009)
)

copyProcess.a9 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerF") ),
  expectedSum = cms.untracked.int32(9012)
)

copyProcess.a10 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerF", processName=cms.InputTag.skipCurrentProcess()) ),
  expectedSum = cms.untracked.int32(6012)
)

copyProcess.a11 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerG") ),
  expectedSum = cms.untracked.int32(9015)
)

copyProcess.a12 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerG", processName=cms.InputTag.skipCurrentProcess()) ),
  expectedSum = cms.untracked.int32(3015)
)

copyProcess.a13 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerH") ),
  expectedSum = cms.untracked.int32(6018)
)

copyProcess.a14 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerH", processName=cms.InputTag.skipCurrentProcess()) ),
  expectedSum = cms.untracked.int32(6018)
)

copyProcess.a15 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerI") ),
  expectedSum = cms.untracked.int32(3021)
)

copyProcess.a16 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerI", processName=cms.InputTag.skipCurrentProcess()) ),
  expectedSum = cms.untracked.int32(3021)
)

copyProcess.a17 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerB", processName=cms.InputTag.currentProcess()) ),
  expectedSum = cms.untracked.int32(9000)
)

copyProcess.a18 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerC", processName=cms.InputTag.currentProcess()) ),
  expectedSum = cms.untracked.int32(9003)
)

copyProcess.a19 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsNotFound = cms.untracked.VInputTag( cms.InputTag("intProducerD", processName=cms.InputTag.currentProcess()) )
)

copyProcess.a20 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsNotFound = cms.untracked.VInputTag( cms.InputTag("intProducerE", processName=cms.InputTag.currentProcess()) )
)

copyProcess.a21 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerF", processName=cms.InputTag.currentProcess()) ),
  expectedSum = cms.untracked.int32(9012)
)

copyProcess.a22 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerG", processName=cms.InputTag.currentProcess()) ),
  expectedSum = cms.untracked.int32(9015)
)

copyProcess.a23 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsNotFound = cms.untracked.VInputTag( cms.InputTag("intProducerH", processName=cms.InputTag.currentProcess()) )
)

copyProcess.a24 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsNotFound = cms.untracked.VInputTag( cms.InputTag("intProducerI", processName=cms.InputTag.currentProcess()) )
)

copyProcess.a25 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(
    cms.InputTag("intProducerB", processName=cms.InputTag.currentProcess()),
    cms.InputTag("intProducerC"),
    cms.InputTag("intProducerD"),
    cms.InputTag("intProducerF", processName=cms.InputTag.currentProcess()),
    cms.InputTag("intProducerG", processName=cms.InputTag.currentProcess()),
    cms.InputTag("intProducerI")
  ),
  inputTagsNotFound = cms.untracked.VInputTag(
    cms.InputTag("intProducerE", processName=cms.InputTag.currentProcess()),
    cms.InputTag("intProducerH", processName=cms.InputTag.currentProcess()),
    cms.InputTag("intProducerQ", "INSTANCE", "DOESNOTEXIST")
  )
)

copyProcess.path2 = cms.Path(
  copyProcess.a1 *
  copyProcess.a2 *
  copyProcess.a3 *
  copyProcess.a4 *
  copyProcess.a5 *
  copyProcess.a6 *
  copyProcess.a7 *
  copyProcess.a8 *
  copyProcess.a9 *
  copyProcess.a10 *
  copyProcess.a11 *
  copyProcess.a12 *
  copyProcess.a13 *
  copyProcess.a14 *
  copyProcess.a15 *
  copyProcess.a16 *
  copyProcess.a17 *
  copyProcess.a18 *
  copyProcess.a19 *
  copyProcess.a20 *
  copyProcess.a21 *
  copyProcess.a22 *
  copyProcess.a23 *
  copyProcess.a24 *
  copyProcess.a25
)

copyProcess.intProducerBeginProcessBlock = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(10001))

copyProcess.intProducerEndProcessBlock = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(100010))

copyProcess.processBlockTest1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  expectedSum = cms.untracked.int32(460034),
  inputTagsBeginProcessBlock = cms.untracked.VInputTag(
    cms.InputTag("intProducerBeginProcessBlock"),
    cms.InputTag("intProducerBeginProcessBlock", "", "PROD1"),
    cms.InputTag("intProducerBeginProcessBlock", "", "COPY")
  ),
  inputTagsEndProcessBlock = cms.untracked.VInputTag(
    cms.InputTag("intProducerBeginProcessBlock"),
    cms.InputTag("intProducerBeginProcessBlock", "", "PROD1"),
    cms.InputTag("intProducerBeginProcessBlock", "", "COPY"),
    cms.InputTag("intProducerEndProcessBlock"),
    cms.InputTag("intProducerEndProcessBlock", "", "PROD1"),
    cms.InputTag("intProducerEndProcessBlock", "", "COPY"),
    cms.InputTag("intProducerEndProcessBlock", "", cms.InputTag.currentProcess())
  )
)

copyProcess.path3 = cms.Path(
    copyProcess.intProducerBeginProcessBlock *
    copyProcess.intProducerEndProcessBlock *
    copyProcess.processBlockTest1
)
