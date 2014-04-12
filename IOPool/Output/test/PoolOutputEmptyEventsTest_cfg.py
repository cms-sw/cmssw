import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTOUTPUT")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.Thing = cms.EDProducer("ThingProducer")

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:PoolOutputEmptyEventsTest.root'),
    SelectEvents = cms.untracked.PSet(SelectEvents=cms.vstring('p'))
)

process.filter = cms.EDFilter("Prescaler",
                              prescaleFactor=cms.int32(100),
                              prescaleOffset=cms.int32(0) )
process.source = cms.Source("EmptySource")

process.p = cms.Path(process.filter+process.Thing*process.OtherThing)
process.ep = cms.EndPath(process.output)


