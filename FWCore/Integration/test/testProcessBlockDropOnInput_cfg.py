import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfThreads = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlockMergeOfMergedFiles.root',
        'file:testProcessBlockMergeOfMergedFiles2.root'
    ),
    inputCommands=cms.untracked.vstring(
        'keep *',
        'drop *_intProducerBeginProcessBlockB_*_*',
        'drop *_intProducerEndProcessBlockB_*_*',
        'drop *_intProducerBeginProcessBlockM_*_*',
        'drop *_intProducerEndProcessBlockM_*_*'
    )
)

process.readProcessBlocksOneAnalyzer1 = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(37),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22, 3300, 4400, 7707),
                                            expectedSum = cms.int32(15440),
                                            consumesProcessBlockNotFound1 = cms.InputTag("intProducerBeginProcessBlockB"),
                                            consumesProcessBlockNotFound2 = cms.InputTag("intProducerEndProcessBlockB"),
                                            consumesProcessBlockNotFound3 = cms.InputTag("intProducerBeginProcessBlockM"),
                                            consumesProcessBlockNotFound4 = cms.InputTag("intProducerEndProcessBlockM")
)

process.readProcessBlocksOneAnalyzer2 = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(28),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlockMM", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlockMM", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(644, 644, 644, 644, 844),
                                            expectedSum = cms.int32(0),
                                            consumesProcessBlockNotFound1 = cms.InputTag("intProducerBeginProcessBlockB"),
                                            consumesProcessBlockNotFound2 = cms.InputTag("intProducerEndProcessBlockB"),
                                            consumesProcessBlockNotFound3 = cms.InputTag("intProducerBeginProcessBlockM"),
                                            consumesProcessBlockNotFound4 = cms.InputTag("intProducerEndProcessBlockM")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockDropOnInput.root'),
        outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_*_*_READ"
    )
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGEOFMERGED'),
    expectedTopProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGEOFMERGED'),
    expectedTopAddedProcesses = cms.untracked.vstring(),
    expectedTopCacheIndices0 = cms.untracked.vuint32(0, 4, 1, 4, 2, 4, 3, 4),
    expectedTopCacheIndices1 = cms.untracked.vuint32(0, 4, 1, 4, 2, 4, 3, 4, 5, 6),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(8),
    expectedProcessesInFirstFile = cms.untracked.uint32(2),
    expectedCacheIndexVectorsPerFile = cms.untracked.vuint32(4),
    expectedNEntries0 = cms.untracked.vuint32(4, 1),
    expectedCacheEntriesPerFile0 =  cms.untracked.vuint32(5)
)


process.p = cms.Path(process.readProcessBlocksOneAnalyzer1 * process.readProcessBlocksOneAnalyzer2)

process.e = cms.EndPath(
    process.out *
    process.testOneOutput
)
