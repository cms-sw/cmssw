import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD1")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(2)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlock2.root'),
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_testEDAliasOriginal_*_*"
    )
)

process.intProducerBeginProcessBlock = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(2))

process.intProducerEndProcessBlock = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(20))

process.intProducerBeginProcessBlockB = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(7))

process.intProducerEndProcessBlockB = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(70))

process.testEDAliasOriginal = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(111))

process.testEDAliasAlias = cms.EDAlias(testEDAliasOriginal = cms.VPSet( cms.PSet(type=cms.string('edmtestIntProduct') ) ) )

process.p = cms.Path(process.intProducerBeginProcessBlock *
                     process.intProducerEndProcessBlock *
                     process.intProducerBeginProcessBlockB *
                     process.intProducerEndProcessBlockB *
                     process.testEDAliasOriginal
)

process.e = cms.EndPath(process.out)
