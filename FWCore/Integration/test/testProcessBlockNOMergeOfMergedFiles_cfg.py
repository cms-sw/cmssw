import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGEOFMERGED")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    fileMode = cms.untracked.string('NOMERGE')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlockMerge.root',
        'file:testProcessBlockMerge2.root'
    )
)

process.readProcessBlocksOneAnalyzer = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(30),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22, 3300, 4400),
                                            expectedSum = cms.int32(8221)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockNOMergeOfMergedFiles.root')
)

process.testGlobalOutput = cms.OutputModule("TestGlobalOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(8)
)

process.testLimitedOutput = cms.OutputModule("TestLimitedOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(8)
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(8),
    testTTreesInFileBlock = cms.untracked.bool(True),
    expectedCacheIndexSize = cms.untracked.vuint32(2, 2, 2, 2, 2, 2, 2, 2)
)

process.intProducerBeginProcessBlockM = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(304))

process.intProducerEndProcessBlockM = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(340))

process.intProducerBeginProcessBlockB = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(308))

process.intProducerEndProcessBlockB = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(380))

process.p = cms.Path(process.intProducerBeginProcessBlockM *
                     process.intProducerEndProcessBlockM *
                     process.intProducerBeginProcessBlockB *
                     process.intProducerEndProcessBlockB *
                     process.readProcessBlocksOneAnalyzer
)

process.e = cms.EndPath(process.out *
                        process.testGlobalOutput *
                        process.testLimitedOutput *
                        process.testOneOutput
)
