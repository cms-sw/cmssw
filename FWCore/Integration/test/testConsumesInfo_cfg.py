import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD1")

process.Tracer = cms.Service('Tracer',
    dumpPathsAndConsumes = cms.untracked.bool(True)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations   = cms.untracked.vstring('cout',
                                           'cerr'
    ),
    categories = cms.untracked.vstring(
        'Tracer'
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet (
            limit = cms.untracked.int32(0)
        ),
        Tracer = cms.untracked.PSet(
            limit=cms.untracked.int32(100000000)
        )
    )
)

process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True),
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
        'drop *_intProducerA_*_*'
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

process.test = cms.EDAnalyzer("TestResultAnalyzer")

process.testView1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsView = cms.untracked.VInputTag( cms.InputTag("intVectorProducer", "", "PROD1") )
)

process.testStreamingProducer = cms.EDProducer("ConsumingIntProducer",
  ivalue = cms.int32(111)
)

process.testStreamingAnalyzer = cms.EDAnalyzer("ConsumingStreamAnalyzer",
  valueMustMatch = cms.untracked.int32(111),
  moduleLabel = cms.untracked.string("testStreamingProducer")
)

process.p = cms.Path(process.intProducer * process.a1 * process.a2 * process.a3 *
                     process.test * process.testView1 *
                     process.testStreamingProducer * process.testStreamingAnalyzer)
process.p2 = cms.Path(process.intProducer * process.a1 * process.a2 * process.a3)
process.p11 = cms.Path()

process.e = cms.EndPath(process.out)
process.p1ep2 = cms.EndPath()

copyProcess = cms.Process("COPY")
process.subProcess = cms.SubProcess(copyProcess,
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_intProducerA_*_*"
    )
)

copyProcess.intVectorProducer = cms.EDProducer("IntVectorProducer",
  count = cms.int32(9),
  ivalue = cms.int32(11)
)

copyProcess.testView1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsView = cms.untracked.VInputTag( cms.InputTag("intVectorProducer", "", "PROD1") )
)

copyProcess.testView2 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsView = cms.untracked.VInputTag( cms.InputTag("intVectorProducer", "", "COPY") )
)

copyProcess.test = cms.EDAnalyzer("TestResultAnalyzer")

copyProcess.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")
copyProcess.testMergeResults = cms.EDAnalyzer("TestMergeResults")

copyProcess.testStreamingProducer = cms.EDProducer("ConsumingIntProducer",
  ivalue = cms.int32(11)
)

copyProcess.testStreamingAnalyzer = cms.EDAnalyzer("ConsumingStreamAnalyzer",
  valueMustMatch = cms.untracked.int32(11),
  moduleLabel = cms.untracked.string("testStreamingProducer")
)

copyProcess.p3 = cms.Path(copyProcess.intVectorProducer * copyProcess.test * copyProcess.thingWithMergeProducer *
                          copyProcess.testMergeResults * copyProcess.testView1 * copyProcess.testView2 *
                          copyProcess.testStreamingProducer * copyProcess.testStreamingAnalyzer)

copyProcess.ep1 = cms.EndPath(copyProcess.intVectorProducer)
copyProcess.ep2 = cms.EndPath()
