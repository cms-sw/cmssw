# This is a test that things will run OK if we disable the
# the strict merging requirement for ProcessBlock products
# in the ProductRegistry merging function. This
# requirement is currently always enforced and this configuration
# will fail.

import FWCore.ParameterSet.Config as cms

process = cms.Process("NONSTRICTTEST")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfThreads = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlockMergeOfMergedFiles.root',
        'file:testProcessBlockMergeOfMergedFiles2.root',
        'file:testProcessBlockDropOnInput.root'
    ),
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_intProducerBeginProcessBlock_*_*',
        'drop *_intProducerEndProcessBlock_*_*',
        'drop *_intProducerBeginProcessBlockMM_*_*',
        'drop *_intProducerEndProcessBlockMM_*_*'
    )
)

# 40 transitions = 30 events + 10 InputProcessBlock transitions + (3 x 0) Cache filling transitions
# sum = 44 + 444 + 44 = 532
process.readProcessBlocksOneAnalyzer1 = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(40),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlock", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlock", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            #expectedByRun = cms.vint32(11, 22, 3300, 4400, 7707),
                                            expectedSum = cms.int32(532)
)

# 40 transitions = 30 events + 10 InputProcessBlock transitions + (3 x 0) Cache filling transitions
# sum = 44 + 444 + 44 = 532
process.readProcessBlocksOneAnalyzer2 = cms.EDAnalyzer("edmtest::one::InputProcessBlockIntAnalyzer",
                                            transitions = cms.int32(40),
                                            consumesBeginProcessBlock = cms.InputTag("intProducerBeginProcessBlockMM", ""),
                                            consumesEndProcessBlock = cms.InputTag("intProducerEndProcessBlockMM", ""),
                                            consumesBeginProcessBlockM = cms.InputTag("intProducerBeginProcessBlockM", ""),
                                            consumesEndProcessBlockM = cms.InputTag("intProducerEndProcessBlockM", ""),
                                            #expectedByRun = cms.vint32(644, 644, 644, 644, 844),
                                            expectedSum = cms.int32(532)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockNonStrict3.root')
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED'),
    expectedTopProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED'),
    expectedTopCacheIndices0 = cms.untracked.vuint32(0, 4, 6, 1, 4, 6, 2, 5, 6, 3, 5, 6),
    expectedTopCacheIndices1 = cms.untracked.vuint32(0, 4, 6, 1, 4, 6, 2, 5, 6, 3, 5, 6, 7, 8, 9),
    expectedTopCacheIndices2 = cms.untracked.vuint32(0, 4, 6, 1, 4, 6, 2, 5, 6, 3, 5, 6, 7, 8, 9, 4294967295, 4294967295, 4294967295),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(11),
    expectedProcessesInFirstFile = cms.untracked.uint32(3),
    expectedCacheIndexVectorsPerFile = cms.untracked.vuint32(4, 1, 1),
    expectedNEntries0 = cms.untracked.vuint32(4, 2, 1),
    expectedNEntries1 = cms.untracked.vuint32(1, 1, 1),
    expectedNEntries2 = cms.untracked.vuint32(0, 0, 0),
    expectedCacheEntriesPerFile0 =  cms.untracked.vuint32(7),
    expectedCacheEntriesPerFile1 =  cms.untracked.vuint32(7, 3),
    expectedCacheEntriesPerFile2 =  cms.untracked.vuint32(7, 3, 0),
    expectedOuterOffset = cms.untracked.vuint32(0, 4, 5)
)

process.p = cms.Path(process.readProcessBlocksOneAnalyzer1 * process.readProcessBlocksOneAnalyzer2)

process.e = cms.EndPath(
    process.out *
    process.testOneOutput
)
