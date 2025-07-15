import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.essource = cms.ESSource("EmptyESSource",
    recordName = cms.string('DummyRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)
process.add_(cms.ESProducer("LoadableDummyProvider",
                            value = cms.untracked.int32(5)))

process.looper = cms.Looper("IntTestLooper",
    expectESValue = cms.untracked.int32(5)
)

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfThreads = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlockMergeOfMergedFiles.root'
    )
)

process.intProducerBeginProcessBlockT = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(4000))

process.intProducerEndProcessBlockT = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(40000))

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockLooperTest.root'),
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_*_beginProcessBlock_*",
        "drop *_*_endProcessBlock_*"
    )
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED', 'TEST'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(24),
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_*_beginProcessBlock_*",
        "drop *_*_endProcessBlock_*"
    )
)

process.eventIntProducer = cms.EDProducer("IntProducer", ivalue = cms.int32(1))

process.transientIntProducerEndProcessBlock = cms.EDProducer("TransientIntProducerEndProcessBlock",
    ivalue = cms.int32(90)
)

process.nonEventIntProducer = cms.EDProducer("NonEventIntProducer",
    ivalue = cms.int32(1)
)

process.p = cms.Path(
    process.eventIntProducer *
    process.transientIntProducerEndProcessBlock *
    process.nonEventIntProducer *
    process.intProducerBeginProcessBlockT *
    process.intProducerEndProcessBlockT
)

process.e = cms.EndPath(
    process.out *
    process.testOneOutput
)
