import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlockMerge.root',
        'file:testProcessBlockMerge2.root'
    )
    , skipEvents = cms.untracked.uint32(7)
)

# 24 transitions = 6 events + 6 access input ProcessBlock transitions + 12 fill calls
# sum 8221 = 3300 + 4400 + 444 + 11 + 22 + 44 = 7700 + 444 + 77 = 
process.readProcessBlocksOneAnalyzer = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(24),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22, 3300, 4400),
                                            expectedSum = cms.int32(8221)
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE'),
    expectedProcessNamesAtWrite = cms.untracked.vstring('PROD1', 'PROD1', 'MERGE', 'PROD1', 'PROD1', 'MERGE', 'TEST'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(7),
    testTTreesInFileBlock = cms.untracked.bool(True),
    expectedCacheIndexSize = cms.untracked.vuint32(2, 2, 2, 4, 4, 4, 4)
)

process.looper = cms.Looper("NavigateEventsLooper")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:testLooperEventNavigation2.root'),
    fastCloning = cms.untracked.bool(False)
)

process.path1 = cms.Path(process.readProcessBlocksOneAnalyzer)
process.endpath1 = cms.EndPath(process.out * process.testOneOutput)
