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
    fileName = cms.untracked.string('testEDAliasTask.root'),
    outputCommands = cms.untracked.vstring(
        'keep *_intProducer_*_*',
    )
)

process.intProducerOrig = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(1))

process.intAlias = cms.EDAlias(
    intProducerOrig = cms.VPSet(cms.PSet(type = cms.string("edmtestIntProduct")))
)

process.intProducer = cms.EDProducer("ManyIntWhenRegisteredProducer", src = cms.string("intAlias"))

process.t = cms.Task(process.intProducer, process.intProducerOrig)
process.p = cms.Path(process.t)

process.e = cms.EndPath(process.out)
