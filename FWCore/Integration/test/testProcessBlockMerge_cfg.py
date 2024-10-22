import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlock1.root',
        'file:testProcessBlock2.root'
    )
)

# transitions 14 = 6 events + 2 InputProcessBlock transitions + 3 x 2 cache filling calls
process.readProcessBlocksOneAnalyzer1 = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(14),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22, 3300, 4400, 7700),
                                            expectedSum = cms.int32(33)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockMerge.root'),
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_testEDAliasAlias_*_*"
    )
)

process.testGlobalOutput = cms.OutputModule("TestGlobalOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(3)
)

process.testLimitedOutput = cms.OutputModule("TestLimitedOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(3)
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE'),
    expectedTopProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE'),
    expectedTopAddedProcesses = cms.untracked.vstring('MERGE'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(3),
    expectedNAddedProcesses = cms.untracked.uint32(1),
    expectedTopCacheIndices0 = cms.untracked.vuint32(0),
    expectedTopCacheIndices1 = cms.untracked.vuint32(0, 1)
)

process.intProducerBeginProcessBlockM = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(4))

process.intProducerEndProcessBlockM = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(40))

process.intProducerBeginProcessBlockB = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(8))

process.intProducerEndProcessBlockB = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(80))

process.a2000 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsInputProcessBlock = cms.untracked.VInputTag( cms.InputTag("testEDAliasAlias")),
  expectedSum = cms.untracked.int32(222),
  expectedCache = cms.untracked.int32(111)
)

process.p = cms.Path(process.intProducerBeginProcessBlockM *
                     process.intProducerEndProcessBlockM *
                     process.intProducerBeginProcessBlockB *
                     process.intProducerEndProcessBlockB *
                     process.readProcessBlocksOneAnalyzer1 *
                     process.a2000
)

process.e = cms.EndPath(process.out *
                        process.testGlobalOutput *
                        process.testLimitedOutput *
                        process.testOneOutput
)
