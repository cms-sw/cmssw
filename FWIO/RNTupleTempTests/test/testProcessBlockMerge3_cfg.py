import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("RNTupleTempSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlockDummy.root'
    )
)

process.out = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string('testProcessBlockMerge3.root')
)

process.testOneOutput = cms.OutputModule("TestOneOutput",
    verbose = cms.untracked.bool(False),
    expectedProcessesWithProcessBlockProducts = cms.untracked.vstring('PROD1', 'MERGE'),
    expectedWriteProcessBlockTransitions = cms.untracked.int32(2)
)

process.intProducerBeginProcessBlockM = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(4))

process.intProducerEndProcessBlockM = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(40))

process.intProducerBeginProcessBlockB = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(8))

process.intProducerEndProcessBlockB = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(80))

process.p = cms.Path(process.intProducerBeginProcessBlockM *
                     process.intProducerEndProcessBlockM *
                     process.intProducerBeginProcessBlockB *
                     process.intProducerEndProcessBlockB
)

process.e = cms.EndPath(process.out *
                        process.testOneOutput
)
