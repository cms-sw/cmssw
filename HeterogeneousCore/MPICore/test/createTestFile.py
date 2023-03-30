import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.options.numberOfStreams = 1
process.options.numberOfThreads = 1

process.maxEvents.input = 20

process.source = cms.Source("EmptySource")

process.things = cms.EDProducer("ThingProducer",
    offsetDelta = cms.int32(100),
    nThings = cms.int32(50)
)

process.path = cms.Path(process.things)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testfile.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_things_*_*',
    )
)

process.end = cms.EndPath(process.out)
