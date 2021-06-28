import FWCore.ParameterSet.Config as cms

process = cms.Process("READSUBPROCESSOUTPUT2")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfThreads = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlockSubProcessReadAgain.root'
    )
)

process.readProcessBlocksOneAnalyzer1 = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(34),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22, 3300, 4400),
                                            expectedSum = cms.int32(8221),
                                            expectedFillerSum = cms.untracked.int32(23199),
                                            consumesBeginProcessBlockNotFound = cms.InputTag("intProducerBeginProcessBlock"),
                                            consumesEndProcessBlockNotFound = cms.InputTag("intProducerEndProcessBlock")
)

process.readProcessBlocksOneAnalyzer2 = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(25),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlockT", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlockT", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(44000, 44000, 44000, 44000),
                                            expectedSum = cms.int32(488),
                                            expectedFillerSum = cms.untracked.int32(132000),
                                            consumesBeginProcessBlockNotFound = cms.InputTag("intProducerBeginProcessBlockT"),
                                            consumesEndProcessBlockNotFound = cms.InputTag("intProducerEndProcessBlockT")
)

process.readProcessBlocksOneAnalyzer3 = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(25),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlockR", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlockR", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(55000, 55000, 55000, 55000),
                                            expectedSum = cms.int32(488),
                                            expectedFillerSum = cms.untracked.int32(165000),
                                            consumesBeginProcessBlockNotFound = cms.InputTag("intProducerBeginProcessBlockR"),
                                            consumesEndProcessBlockNotFound = cms.InputTag("intProducerEndProcessBlockR")
)

process.readProcessBlocksOneAnalyzer4 = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(25),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlockRA", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlockRA", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(1100000, 1100000, 1100000, 1100000),
                                            expectedSum = cms.int32(488),
                                            expectedFillerSum = cms.untracked.int32(3300000),
                                            consumesBeginProcessBlockNotFound = cms.InputTag("intProducerBeginProcessBlockRA"),
                                            consumesEndProcessBlockNotFound = cms.InputTag("intProducerEndProcessBlockRA")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockSubProcessRead2.root')
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED', 'TEST', 'READ', 'READAGAIN'),
    expectedTopProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED', 'TEST', 'READ', 'READAGAIN'),
    expectedTopAddedProcesses = cms.untracked.vstring(),
    expectedTopCacheIndices0 = cms.untracked.vuint32(0, 4, 6, 7, 8, 9, 1, 4, 6, 7, 8, 9, 2, 5, 6, 7, 8, 9, 3, 5, 6, 7, 8, 9),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(11)
)

process.p = cms.Path(
    process.readProcessBlocksOneAnalyzer1 *
    process.readProcessBlocksOneAnalyzer2 *
    process.readProcessBlocksOneAnalyzer3 *
    process.readProcessBlocksOneAnalyzer4
)

process.e = cms.EndPath(
    process.out *
    process.testOneOutput
)
