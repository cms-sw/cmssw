import FWCore.ParameterSet.Config as cms

# Intentionally reversing the order of process names in this series
process = cms.Process("PROD1")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlockMerge3.root'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockMergeOfMergedFiles2.root')
)

process.intProducerBeginProcessBlock = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(707))

process.intProducerEndProcessBlock = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(7000))

process.intProducerBeginProcessBlockB = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(40000))

process.intProducerEndProcessBlockB = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(400000))

process.p = cms.Path(process.intProducerBeginProcessBlock *
                     process.intProducerEndProcessBlock *
                     process.intProducerBeginProcessBlockB *
                     process.intProducerEndProcessBlockB
)

process.e = cms.EndPath(process.out)
