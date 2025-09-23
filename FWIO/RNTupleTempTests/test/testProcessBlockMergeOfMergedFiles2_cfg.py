import FWCore.ParameterSet.Config as cms

# Intentionally reversing the order of process names in this series
process = cms.Process("MERGEOFMERGED")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("RNTupleTempSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlockMerge3.root'
    )
)

process.out = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string('testProcessBlockMergeOfMergedFiles2.root')
)

#NEW
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
