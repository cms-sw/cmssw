import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlockDropOnOutput.root'
    )
)

# 37 = 15 events + 7 access input ProcessBlock transitions + 15 fillCache functor calls
# sum 15440 = 11 + 22 + 3300 + 4400 + 7707
process.readProcessBlocksOneAnalyzer = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(37),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22, 3300, 4400, 7707),
                                            expectedSum = cms.int32(15440)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockReadDropOnOutput.root')
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGEOFMERGED'),
    expectedTopProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGEOFMERGED'),
    expectedTopAddedProcesses = cms.untracked.vstring(),
    expectedTopCacheIndices0 = cms.untracked.vuint32(0, 5, 1, 5, 2, 5, 3, 5, 4, 6),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(8)
)


process.p = cms.Path(process.readProcessBlocksOneAnalyzer)

process.e = cms.EndPath(
    process.out *
    process.testOneOutput
)
