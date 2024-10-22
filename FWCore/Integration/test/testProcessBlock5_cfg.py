import FWCore.ParameterSet.Config as cms

# Intentionally reversing the order of process names in this series
process = cms.Process("MERGEOFMERGED")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(5)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProcessBlock5.root')
)


process.intProducerBeginProcessBlockMM = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(404))

process.intProducerEndProcessBlockMM = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(440))

process.intProducerBeginProcessBlockB = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(308))

process.intProducerEndProcessBlockB = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(380))

process.p = cms.Path(process.intProducerBeginProcessBlockMM *
                     process.intProducerEndProcessBlockMM *
                     process.intProducerBeginProcessBlockB *
                     process.intProducerEndProcessBlockB
)

process.e = cms.EndPath(process.out)
