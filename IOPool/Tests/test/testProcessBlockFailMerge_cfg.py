import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

# The second file has ProcessBlock products dropped that
# were in the first file so this should fail the strict
# merge requirements on ProcessBlock products.
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlock1.root',
        'file:testProcessBlock2Dropped.root'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlockFailMerge.root')
)

process.intProducerBeginProcessBlock = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(2))

process.intProducerEndProcessBlock = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(20))

process.p = cms.Path(process.intProducerBeginProcessBlock *
                     process.intProducerEndProcessBlock)

process.e = cms.EndPath(process.out)
