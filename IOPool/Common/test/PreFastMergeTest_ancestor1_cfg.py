import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTPROD")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.Thing = cms.EDProducer("ThingProducer")
process.AThing = cms.EDProducer("ThingProducer")
process.ZThing = cms.EDProducer("ThingProducer")

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.AOtherThing = cms.EDProducer("OtherThingProducer",
    thingTag = cms.InputTag('AThing')
)
process.ZOtherThing = cms.EDProducer("OtherThingProducer",
    thingTag = cms.InputTag('ZThing')
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:FastMergeTest_ancestor1.root'),
    outputCommands = cms.untracked.vstring('keep *', 
        'drop *_Thing_*_*',
        'drop *_AThing_*_*',
        'drop *_ZThing_*_*')
)

process.source = cms.Source("EmptySource",
    firstEvent = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(100)
)

process.p = cms.Path(process.Thing*process.OtherThing+process.AThing*process.AOtherThing+process.ZThing*process.ZOtherThing)
process.ep = cms.EndPath(process.output)


