import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD1")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlock1.root'),
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_testEDAliasOriginal_*_*"
    )
)

process.testGlobalOutput = cms.OutputModule("TestGlobalOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(1),
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_testEDAliasOriginal_*_*"
    )
)

process.testLimitedOutput = cms.OutputModule("TestLimitedOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(1),
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_testEDAliasOriginal_*_*"
    )
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1'),
    expectedTopProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(1),
    requireNullTTreesInFileBlock = cms.untracked.bool(True),
    expectedProductsFromInputKept = cms.untracked.bool(False),
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_testEDAliasOriginal_*_*"
    )
)

process.intProducerBeginProcessBlock = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(1))

process.intProducerEndProcessBlock = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(10))

process.intProducerBeginProcessBlockB = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(5))

process.intProducerEndProcessBlockB = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(50))

process.testEDAliasOriginal = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(111))

process.testEDAliasAlias = cms.EDAlias(testEDAliasOriginal = cms.VPSet( cms.PSet(type=cms.string('edmtestIntProduct') ) ) )

process.a1 = cms.EDAnalyzer("TestFindProduct",
   expectedSum = cms.untracked.int32(111),
   inputTags = cms.untracked.VInputTag(),
   inputTagsBeginProcessBlock = cms.untracked.VInputTag(
     cms.InputTag("testEDAliasOriginal")
   )
)

process.a2 = cms.EDAnalyzer("TestFindProduct",
   expectedSum = cms.untracked.int32(111),
   inputTags = cms.untracked.VInputTag(),
   inputTagsBeginProcessBlock = cms.untracked.VInputTag(
     cms.InputTag("testEDAliasAlias")
   )
)

process.p = cms.Path(process.intProducerBeginProcessBlock *
                     process.intProducerEndProcessBlock *
                     process.intProducerBeginProcessBlockB *
                     process.intProducerEndProcessBlockB *
                     process.testEDAliasOriginal *
                     process.a1 *
                     process.a2
)

process.e = cms.EndPath(process.out *
                        process.testGlobalOutput *
                        process.testLimitedOutput *
                        process.testOneOutput
)
