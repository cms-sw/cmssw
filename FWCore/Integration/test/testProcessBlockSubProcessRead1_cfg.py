import FWCore.ParameterSet.Config as cms

process = cms.Process("READSUBPROCESSOUTPUT")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfThreads = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlockSubProcessRead.root'
    )
)

process.readProcessBlocksOneAnalyzer1 = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(33),
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
                                            transitions = cms.int32(24),
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
                                            transitions = cms.int32(24),
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

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockSubProcessRead1.root')
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED', 'TEST', 'READ'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(10)
)

process.p = cms.Path(
    process.readProcessBlocksOneAnalyzer1 *
    process.readProcessBlocksOneAnalyzer2 *
    process.readProcessBlocksOneAnalyzer3
)

process.e = cms.EndPath(
    process.out *
    process.testOneOutput
)
