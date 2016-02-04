import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTPROD")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.Thing = cms.EDProducer("ThingProducer",
    debugLevel = cms.untracked.int32(0)
)

process.OtherThing = cms.EDProducer("OtherThingProducer",
    debugLevel = cms.untracked.int32(0)
)

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *', 
        'drop *_OtherThing_*_*'),
    fileName = cms.untracked.string('file:FastMergeTest_1.root')
)

process.source = cms.Source("EmptySource",
    firstEvent = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(100)
)

process.p = cms.Path(process.Thing*process.OtherThing)
process.ep = cms.EndPath(process.output)


