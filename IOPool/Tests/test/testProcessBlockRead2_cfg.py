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
        'file:testProcessBlockMergeOfMergedFiles.root'
    )
)

process.readProcessBlocksOneAnalyzer = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(31),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22, 3300, 4400),
                                            expectedSum = cms.int32(8221)
)

process.transientIntProducerEndProcessBlock = cms.EDProducer("TransientIntProducerEndProcessBlock",
    ivalue = cms.int32(90)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockRead2.root')
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED'),
    expectedTopProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED'),
    expectedTopAddedProcesses = cms.untracked.vstring(),
    expectedTopCacheIndices0 = cms.untracked.vuint32(0, 4, 6, 1, 4, 6, 2, 5, 6, 3, 5, 6),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(8)
)


process.p = cms.Path(process.transientIntProducerEndProcessBlock * process.readProcessBlocksOneAnalyzer)

process.e = cms.EndPath(
    process.out *
    process.testOneOutput
)
