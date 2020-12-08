import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD1")

process.Tracer = cms.Service('Tracer',
                             dumpContextForLabels = cms.untracked.vstring('intProducerA', 'a1'),
                             dumpNonModuleContext = cms.untracked.bool(True)
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
    fileName = cms.untracked.string('testGetBy1.root'),
    outputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_intProducerA_*_*'
    )
)

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("source") ),
  expectedSum = cms.untracked.int32(530021),
  inputTagsNotFound = cms.untracked.VInputTag(
    cms.InputTag("source", processName=cms.InputTag.skipCurrentProcess()),
    cms.InputTag("intProducer", processName=cms.InputTag.skipCurrentProcess()),
    cms.InputTag("intProducerU", processName=cms.InputTag.skipCurrentProcess())
  ),
  inputTagsBeginProcessBlock = cms.untracked.VInputTag(
    cms.InputTag("intProducerBeginProcessBlock"),
  ),
  inputTagsEndProcessBlock = cms.untracked.VInputTag(
    cms.InputTag("intProducerEndProcessBlock"),
  ),
  inputTagsEndProcessBlock2 = cms.untracked.VInputTag(
    cms.InputTag("intProducerEndProcessBlock", "two"),
  ),
  inputTagsEndProcessBlock3 = cms.untracked.VInputTag(
    cms.InputTag("intProducerEndProcessBlock", "three"),
  ),
  inputTagsEndProcessBlock4 = cms.untracked.VInputTag(
    cms.InputTag("intProducerEndProcessBlock", "four"),
  ),
  testGetterOfProducts = cms.untracked.bool(True)
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

process.intProducerB = cms.EDProducer("IntProducer", ivalue = cms.int32(1000))

process.intProducerBeginProcessBlock = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(10000))

process.intProducerEndProcessBlock = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(100000))

process.t = cms.Task(process.intProducerU,
                     process.intProducerA,
                     process.intProducerB,
                     process.intVectorProducer,
                     process.intProducerBeginProcessBlock,
                     process.intProducerEndProcessBlock
)

process.p = cms.Path(process.intProducer * process.a1 * process.a2 * process.a3, process.t)

process.e = cms.EndPath(process.out)

copyProcess = cms.Process("COPY")
process.addSubProcess(cms.SubProcess(copyProcess,
    outputCommands = cms.untracked.vstring(
        "keep *", 
        "drop *_intProducerA_*_*"
    )
))
