import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGEOFMERGED")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("RNTupleTempSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlockMerge.root',
        'file:testProcessBlockMerge2.root'
    )
)

# In the next module, the expectedSum is a sum of values from previous
# processes as follow:
#
#     testProcessBlock1_cfg.py      intProducerBeginProcessBlock     1
#                                   intProducerEndProcessBlock      10
#     testProcessBlock2_cfg.py      intProducerBeginProcessBlock     2
#                                   intProducerEndProcessBlock      20
#     testProcessBlock3_cfg.py      intProducerBeginProcessBlock   300
#                                   intProducerEndProcessBlock    3000
#     testProcessBlock4_cfg.py      intProducerBeginProcessBlock   400
#                                   intProducerEndProcessBlock    4000
#     testProcessBlockMerge_cfg.py  intProducerBeginProcessBlockM    4
#                                   intProducerBeginProcessBlockM   40
#     testProcessBlockMerge2_cfg.py  intProducerBeginProcessBlockM  204
#                                   intProducerBeginProcessBlockM  240
#
#     TOTAL                                                       8221

process.readProcessBlocksOneAnalyzer = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(30),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            expectedByRun = cms.vint32(11, 22, 3300, 4400),
                                            expectedSum = cms.int32(8221)
)

process.out = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string('testProcessBlockMergeOfMergedFiles.root')
)

process.testGlobalOutput = cms.OutputModule("TestGlobalOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(7)
)

process.testLimitedOutput = cms.OutputModule("TestLimitedOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(7)
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED'),
    expectedTopProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED'),
    expectedTopAddedProcesses = cms.untracked.vstring('MERGEOFMERGED'),
    expectedProcessNamesAtWrite = cms.untracked.vstring('PROD1', 'PROD1', 'MERGE', 'PROD1', 'PROD1', 'MERGE', 'MERGEOFMERGED'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(7),
    testTTreesInFileBlock = cms.untracked.bool(True),
    expectedCacheIndexSize = cms.untracked.vuint32(2, 2, 2, 4, 4, 4, 4),
    expectedNAddedProcesses = cms.untracked.uint32(1),
    expectedTopCacheIndices0 = cms.untracked.vuint32(0, 2, 1, 2),
    expectedTopCacheIndices1 = cms.untracked.vuint32(0, 2, 1, 2, 3, 5, 4, 5)
)

process.intProducerBeginProcessBlockMM = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(304))

process.intProducerEndProcessBlockMM = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(340))

process.intProducerBeginProcessBlockB = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(308))

process.intProducerEndProcessBlockB = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(380))

process.p = cms.Path(process.intProducerBeginProcessBlockMM *
                     process.intProducerEndProcessBlockMM *
                     process.intProducerBeginProcessBlockB *
                     process.intProducerEndProcessBlockB *
                     process.readProcessBlocksOneAnalyzer
)

process.e = cms.EndPath(process.out *
                        process.testGlobalOutput *
                        process.testLimitedOutput *
                        process.testOneOutput
)
