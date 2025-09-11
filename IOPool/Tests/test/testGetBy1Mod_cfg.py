import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD1")

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(2)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGetBy1Mod.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.intProducer = cms.EDProducer("IntProducer", ivalue = cms.int32(2))

process.intProducerBeginProcessBlock = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(10000))

process.intProducerEndProcessBlock = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(100000))

process.t = cms.Task(process.intProducer,
                     process.intProducerBeginProcessBlock,
                     process.intProducerEndProcessBlock )

process.e = cms.EndPath(process.out, process.t)
