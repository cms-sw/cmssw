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
        'file:testProcessBlockTest.root'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockRead.root')
)

process.testGlobalOutput = cms.OutputModule("TestGlobalOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'READ'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(4)
)

process.testLimitedOutput = cms.OutputModule("TestLimitedOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'READ'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(4)
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'READ'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(4)
)

process.intProducerBeginProcessBlockR = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(100))

process.intProducerEndProcessBlockR = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(1000))

process.readProcessBlocks = cms.EDAnalyzer("edmtest::stream::InputProcessBlockIntAnalyzer",
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            sleepTime = cms.uint32(10000)
)

process.readProcessBlocksG = cms.EDAnalyzer("edmtest::stream::InputProcessBlockIntAnalyzerG",
                                            transitions = cms.int32(15),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            expectedSum = cms.int32(77),
                                            sleepTime = cms.uint32(10000)
)

process.readProcessBlocksNS = cms.EDAnalyzer("edmtest::stream::InputProcessBlockIntAnalyzerNS")
process.readProcessBlocksGNS = cms.EDAnalyzer("edmtest::stream::InputProcessBlockIntAnalyzerGNS")

process.readProcessBlocksStreamFilter = cms.EDFilter("edmtest::stream::InputProcessBlockIntFilter",
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            sleepTime = cms.uint32(10000)
)

process.readProcessBlocksGStreamFilter = cms.EDFilter("edmtest::stream::InputProcessBlockIntFilterG",
                                            transitions = cms.int32(15),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            expectedSum = cms.int32(77),
                                            sleepTime = cms.uint32(10000)
)

process.readProcessBlocksStreamProducer = cms.EDProducer("edmtest::stream::InputProcessBlockIntProducer",
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            sleepTime = cms.uint32(10000)
)

process.readProcessBlocksGStreamProducer = cms.EDProducer("edmtest::stream::InputProcessBlockIntProducerG",
                                            transitions = cms.int32(15),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            expectedSum = cms.int32(77),
                                            sleepTime = cms.uint32(10000)
)

process.readProcessBlocksOneAnalyzer = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(15),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            expectedSum = cms.int32(77),
                                            consumesBeginProcessBlockNotFound = cms.InputTag("intProducerBeginProcessBlock"),
                                            consumesEndProcessBlockNotFound = cms.InputTag("intProducerEndProcessBlock")
)

process.readProcessBlocksOneFilter = cms.EDFilter("edmtest::one::InputProcessBlockIntFilter",
                                            transitions = cms.int32(15),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            expectedSum = cms.int32(77)
)

process.readProcessBlocksOneProducer = cms.EDProducer("edmtest::one::InputProcessBlockIntProducer",
                                            transitions = cms.int32(15),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            expectedSum = cms.int32(77)
)

process.readProcessBlocksGlobalAnalyzer = cms.EDAnalyzer("edmtest::global::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(15),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            expectedSum = cms.int32(77)
)

process.readProcessBlocksGlobalFilter = cms.EDFilter("edmtest::global::InputProcessBlockIntFilter",
                                            transitions = cms.int32(15),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            expectedSum = cms.int32(77)
)

process.readProcessBlocksGlobalProducer = cms.EDProducer("edmtest::global::InputProcessBlockIntProducer",
                                            transitions = cms.int32(15),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            expectedSum = cms.int32(77)
)

process.readProcessBlocksLimitedAnalyzer = cms.EDAnalyzer("edmtest::limited::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(15),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            expectedSum = cms.int32(77),
                                            concurrencyLimit = cms.untracked.uint32(4)
)

process.readProcessBlocksLimitedFilter = cms.EDFilter("edmtest::limited::InputProcessBlockIntFilter",
                                            transitions = cms.int32(15),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            expectedSum = cms.int32(77),
                                            concurrencyLimit = cms.untracked.uint32(4)
)

process.readProcessBlocksLimitedProducer = cms.EDProducer("edmtest::limited::InputProcessBlockIntProducer",
                                            transitions = cms.int32(15),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22),
                                            expectedSum = cms.int32(77),
                                            concurrencyLimit = cms.untracked.uint32(4)
)

process.readProcessBlocksGlobalAnalyzerNoRegistration = cms.EDAnalyzer("edmtest::global::InputProcessBlockIntAnalyzerNoRegistration",
                                            transitions = cms.int32(6),
)

process.readProcessBlocksDoesNotExist = cms.EDAnalyzer("edmtest::global::InputProcessBlockAnalyzerThreeTags",
                                            transitions = cms.int32(9),
                                            consumesBeginProcessBlock0 = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock0 = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlock1 = cms.InputTag("doesNotExist", ""),
                                            consumesEndProcessBlock1 = cms.InputTag("doesNotExist", ""),
                                            consumesBeginProcessBlock2 = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlock2 = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun0 = cms.vint32(11, 22),
                                            expectedByRun1 = cms.vint32(),
                                            expectedByRun2 = cms.vint32(44, 44)
)

process.readProcessBlocksExplicitProcess = cms.EDAnalyzer("edmtest::global::InputProcessBlockAnalyzerThreeTags",
                                            transitions = cms.int32(10),
                                            consumesBeginProcessBlock0 = cms.InputTag("intProducerBeginProcessBlockB", ""),
                                            consumesEndProcessBlock0 = cms.InputTag("intProducerEndProcessBlockB", ""),
                                            consumesBeginProcessBlock1 = cms.InputTag("intProducerBeginProcessBlockB", "", "PROD1"),
                                            consumesEndProcessBlock1 = cms.InputTag("intProducerEndProcessBlockB", "", "PROD1"),
                                            consumesBeginProcessBlock2 = cms.InputTag("intProducerBeginProcessBlockB", "", "MERGE"),
                                            consumesEndProcessBlock2 = cms.InputTag("intProducerEndProcessBlockB", "", "MERGE"),
                                            expectedByRun0 = cms.vint32(88, 88),
                                            expectedByRun1 = cms.vint32(55, 77),
                                            expectedByRun2 = cms.vint32(88, 88)
)

process.readProcessBlocksExplicitProcess2 = cms.EDAnalyzer("edmtest::global::InputProcessBlockAnalyzerThreeTags",
                                            transitions = cms.int32(11),
                                            consumesBeginProcessBlock0 = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock0 = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlock1 = cms.InputTag("intProducerBeginProcessBlock", "", "PROD1"),
                                            consumesEndProcessBlock1 = cms.InputTag("intProducerEndProcessBlock", "", "PROD1"),
                                            consumesBeginProcessBlock2 = cms.InputTag("intProducerBeginProcessBlockM", "", "MERGE"),
                                            consumesEndProcessBlock2 = cms.InputTag("intProducerEndProcessBlockM", "", "MERGE"),
                                            expectedByRun0 = cms.vint32(11, 22),
                                            expectedByRun1 = cms.vint32(11, 22),
                                            expectedByRun2 = cms.vint32(44, 44)
)

process.readProcessBlocksReuseCache = cms.EDAnalyzer("edmtest::global::InputProcessBlockAnalyzerReuseCache",
                                            transitions = cms.int32(8),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            expectedByRun = cms.vint32(11, 11)
)

process.p = cms.Path(process.intProducerBeginProcessBlockR *
                     process.intProducerEndProcessBlockR *
                     process.readProcessBlocks *
                     process.readProcessBlocksG *
                     process.readProcessBlocksNS *
                     process.readProcessBlocksGNS *
                     process.readProcessBlocksStreamFilter *
                     process.readProcessBlocksGStreamFilter *
                     process.readProcessBlocksStreamProducer *
                     process.readProcessBlocksGStreamProducer *
                     process.readProcessBlocksOneAnalyzer *
                     process.readProcessBlocksOneFilter *
                     process.readProcessBlocksOneProducer *
                     process.readProcessBlocksGlobalAnalyzer *
                     process.readProcessBlocksGlobalFilter *
                     process.readProcessBlocksGlobalProducer *
                     process.readProcessBlocksLimitedAnalyzer *
                     process.readProcessBlocksLimitedFilter *
                     process.readProcessBlocksLimitedProducer *
                     process.readProcessBlocksDoesNotExist *
                     process.readProcessBlocksGlobalAnalyzerNoRegistration *
                     process.readProcessBlocksExplicitProcess *
                     process.readProcessBlocksExplicitProcess2 *
                     process.readProcessBlocksReuseCache
)

process.e = cms.EndPath(
    process.out *
    process.testGlobalOutput *
    process.testLimitedOutput *
    process.testOneOutput
)
