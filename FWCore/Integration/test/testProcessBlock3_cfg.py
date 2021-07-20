import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD1")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(3)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlock3.root')
)

process.intProducerBeginProcessBlock = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(300))

process.intProducerEndProcessBlock = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(3000))

process.intProducerBeginProcessBlockB = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(30000))

process.intProducerEndProcessBlockB = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(300000))

process.p = cms.Path(process.intProducerBeginProcessBlock *
                     process.intProducerEndProcessBlock *
                     process.intProducerBeginProcessBlockB *
                     process.intProducerEndProcessBlockB
)

process.e = cms.EndPath(process.out)
