import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlockDropOnOutput2.root'
    )
)

# 38 = 15 events + 8 access input ProcessBlock transitions + 15 fillCache functor calls
# sum 15440 = 11 + 22 + 3300 + 4400 + 7707
process.readProcessBlocksOneAnalyzer = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(38),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22, 3300, 4400, 7707),
                                            expectedSum = cms.int32(15440)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockReadDropOnOutput2.root')
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGEOFMERGED', 'DROPONOUTPUT'),
    expectedTopProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGEOFMERGED', 'DROPONOUTPUT'),
    expectedTopAddedProcesses = cms.untracked.vstring(),
    expectedTopCacheIndices0 = cms.untracked.vuint32(0, 5, 7, 1, 5, 7, 2, 5, 7, 3, 5, 7, 4, 6, 7),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(9)
)


process.p = cms.Path(process.readProcessBlocksOneAnalyzer)

process.e = cms.EndPath(
    process.out *
    process.testOneOutput
)
