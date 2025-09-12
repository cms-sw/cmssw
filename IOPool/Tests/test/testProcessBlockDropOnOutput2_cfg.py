import FWCore.ParameterSet.Config as cms

process = cms.Process("DROPONOUTPUT")

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
)

process.intProducerBeginProcessBlockN = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(5))

process.intProducerEndProcessBlockN = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(50))

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockDropOnOutput2.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_intProducerBeginProcessBlockB_*_*',
        'drop *_intProducerEndProcessBlockB_*_*',
        'drop *_intProducerBeginProcessBlockM_*_*',
        'drop *_intProducerEndProcessBlockM_*_*'
    )
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_intProducerBeginProcessBlockB_*_*',
        'drop *_intProducerEndProcessBlockB_*_*',
        'drop *_intProducerBeginProcessBlockM_*_*',
        'drop *_intProducerEndProcessBlockM_*_*'
    ),
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGEOFMERGED', 'DROPONOUTPUT'),
    expectedTopProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED', 'DROPONOUTPUT'),
    expectedTranslateFromStoredIndex = cms.untracked.vuint32(0, 2, 3),
    expectedNAddedProcesses = cms.untracked.uint32(1),
    expectedProductsFromInputKept = cms.untracked.bool(True)
)

process.out2 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockDropOnOutput2_2.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_intProducerBeginProcessBlock_*_*',
        'drop *_intProducerEndProcessBlock_*_*',
        'drop *_intProducerBeginProcessBlockB_*_*',
        'drop *_intProducerEndProcessBlockB_*_*',
        'drop *_intProducerBeginProcessBlockM_*_*',
        'drop *_intProducerEndProcessBlockM_*_*',
        'drop *_intProducerBeginProcessBlockMM_*_*',
        'drop *_intProducerEndProcessBlockMM_*_*'
    )
)

process.testOneOutput2 = cms.OutputModule("TestOneOutput",
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_intProducerBeginProcessBlock_*_*',
        'drop *_intProducerEndProcessBlock_*_*',
        'drop *_intProducerBeginProcessBlockB_*_*',
        'drop *_intProducerEndProcessBlockB_*_*',
        'drop *_intProducerBeginProcessBlockM_*_*',
        'drop *_intProducerEndProcessBlockM_*_*',
        'drop *_intProducerBeginProcessBlockMM_*_*',
        'drop *_intProducerEndProcessBlockMM_*_*'
    ),
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('DROPONOUTPUT'),
    expectedTopProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE', 'MERGEOFMERGED', 'DROPONOUTPUT'),
    expectedTranslateFromStoredIndex = cms.untracked.vuint32(3),
    expectedNAddedProcesses = cms.untracked.uint32(1),
    expectedProductsFromInputKept = cms.untracked.bool(False)
)

process.p = cms.Path(process.intProducerBeginProcessBlockN * process.intProducerEndProcessBlockN)

process.e = cms.EndPath(
    process.out *
    process.testOneOutput *
    process.out2 *
    process.testOneOutput2
)
