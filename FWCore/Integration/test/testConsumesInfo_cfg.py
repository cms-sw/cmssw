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

process.runLumiESSource = cms.ESSource("RunLumiESSource")

process.testReadLumiESSource = cms.EDAnalyzer("RunLumiESAnalyzer")

process.testReadLumiESSource1 = cms.EDAnalyzer("RunLumiESAnalyzer",
    esInputTag = cms.ESInputTag('runLumiESSource', ''),
    getIntProduct = cms.bool(True)
)

process.testReadLumiESSource2 = cms.EDAnalyzer("RunLumiESAnalyzer",
    esInputTag = cms.ESInputTag('runLumiESSource', 'productLabelThatDoesNotExist'),
    checkDataProductContents = cms.bool(False)
)

process.testReadLumiESSource3 = cms.EDAnalyzer("RunLumiESAnalyzer",
    esInputTag = cms.ESInputTag('moduleLabelThatDoesNotMatch', ''),
    checkDataProductContents = cms.bool(False)
)

process.concurrentIOVESSource = cms.ESSource("ConcurrentIOVESSource",
    iovIsRunNotTime = cms.bool(True),
    firstValidLumis = cms.vuint32(1, 4, 6, 7, 8, 9),
    invalidLumis = cms.vuint32(),
    concurrentFinder = cms.bool(True)
)

process.concurrentIOVESProducer = cms.ESProducer("ConcurrentIOVESProducer")

process.concurrentIOVAnalyzer = cms.EDAnalyzer("ConcurrentIOVAnalyzer",
                              checkExpectedValues = cms.untracked.bool(False)
)

process.WhatsItAnalyzer = cms.EDAnalyzer("WhatsItAnalyzer",
    expectedValues = cms.untracked.vint32(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))


process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

process.consumeWhatsIt = cms.ESProducer("ConsumeWhatsIt",
    esInputTag_in_produce = cms.ESInputTag("", ""),
    esInputTagA_in_produce = cms.ESInputTag("", "A"),
    esInputTagB_in_produce = cms.ESInputTag("", "B"),
    esInputTagC_in_produce = cms.ESInputTag("", "C"),
    esInputTagD_in_produce = cms.ESInputTag("", "D"),
    esInputTag_in_produceA = cms.ESInputTag("", ""),
    esInputTagA_in_produceA = cms.ESInputTag("", "A"),
    esInputTagB_in_produceA = cms.ESInputTag("", "B"),
    esInputTagC_in_produceA = cms.ESInputTag("", "C"),
    esInputTagD_in_produceA = cms.ESInputTag("", "D"),
    esInputTag_in_produceB = cms.ESInputTag("", ""),
    esInputTagA_in_produceB = cms.ESInputTag("", "A"),
    esInputTagB_in_produceB = cms.ESInputTag("", "B"),
    esInputTagC_in_produceB = cms.ESInputTag("", "C"),
    esInputTagD_in_produceB = cms.ESInputTag("", "D"),
    esInputTag_in_produceC = cms.ESInputTag("", "productLabelThatDoesNotExist"),
    esInputTagA_in_produceC = cms.ESInputTag("", "productLabelThatDoesNotExist"),
    esInputTagB_in_produceC = cms.ESInputTag("moduleLabelThatDoesNotMatch", "B"),
    esInputTagC_in_produceC = cms.ESInputTag("moduleLabelThatDoesNotMatch", "C"),
    esInputTagD_in_produceC = cms.ESInputTag("moduleLabelThatDoesNotMatch", "D"),
    esInputTag_in_produceD = cms.ESInputTag("WhatsItESProducer", ""),
    esInputTagA_in_produceD = cms.ESInputTag("WhatsItESProducer", "A"),
    esInputTagB_in_produceD = cms.ESInputTag("WhatsItESProducer", "B"),
    esInputTagC_in_produceD = cms.ESInputTag("WhatsItESProducer", "C"),
    esInputTagD_in_produceD = cms.ESInputTag("WhatsItESProducer", "D")
)

process.mayConsumeWhatsIt = cms.ESProducer("MayConsumeWhatsIt")

process.consumeIOVTestInfoAnalyzer = cms.EDAnalyzer("ConsumeIOVTestInfoAnalyzer",
    esInputTag = cms.untracked.ESInputTag("", "DependsOnMayConsume")
)

process.consumeIOVTestInfoAnalyzer2 = cms.EDAnalyzer("ConsumeIOVTestInfoAnalyzer",
    esInputTag = cms.untracked.ESInputTag("", "DependsOnMayConsume2")
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

process.testEventSetupPath = cms.Path(
    process.testReadLumiESSource *
    process.testReadLumiESSource1 *
    process.testReadLumiESSource2 *
    process.testReadLumiESSource3 *
    process.concurrentIOVAnalyzer *
    process.WhatsItAnalyzer *
    process.consumeIOVTestInfoAnalyzer *
    process.consumeIOVTestInfoAnalyzer2
)

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
