import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTOUTPUT")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.options.allowUnscheduled = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.Thing = cms.EDProducer("ThingProducer",
    debugLevel = cms.untracked.int32(1)
)

process.OtherThing = cms.EDProducer("OtherThingProducer",
    debugLevel = cms.untracked.int32(1)
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:PoolOutputTestUnscheduled.root')
)

process.source = cms.Source("EmptySource")

process.ep = cms.EndPath(process.output)


