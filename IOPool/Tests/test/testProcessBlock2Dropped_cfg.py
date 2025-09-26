import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD1")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(2),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(100)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('testProcessBlock2Dropped.root'),
  outputCommands = cms.untracked.vstring(
    'keep *',
    'drop *_intProducerBeginProcessBlock_*_*',
    'drop *_intProducerEndProcessBlock_*_*'
  )
)

process.testGlobalOutput = cms.OutputModule("TestGlobalOutput",
    verbose = cms.untracked.bool(False),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(1),
    outputCommands = cms.untracked.vstring(
      'keep *',
      'drop *_intProducerBeginProcessBlock_*_*',
      'drop *_intProducerEndProcessBlock_*_*'
    )
)

process.testLimitedOutput = cms.OutputModule("TestLimitedOutput",
    verbose = cms.untracked.bool(False),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(1),
    outputCommands = cms.untracked.vstring(
      'keep *',
      'drop *_intProducerBeginProcessBlock_*_*',
      'drop *_intProducerEndProcessBlock_*_*'
    )
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(1),
    outputCommands = cms.untracked.vstring(
      'keep *',
      'drop *_intProducerBeginProcessBlock_*_*',
      'drop *_intProducerEndProcessBlock_*_*'
    ),
    expectedProductsFromInputKept = cms.untracked.bool(False)
)

process.intProducerBeginProcessBlock = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(1))

process.intProducerEndProcessBlock = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(10))

process.p = cms.Path(process.intProducerBeginProcessBlock *
                     process.intProducerEndProcessBlock)

process.e = cms.EndPath(process.out *
                        process.testGlobalOutput *
                        process.testLimitedOutput *
                        process.testOneOutput
)
