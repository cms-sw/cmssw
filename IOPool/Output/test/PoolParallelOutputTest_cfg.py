import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTOUTPUT")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.Thing = cms.EDProducer("ThingProducer")

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:PoolOutputTest.root')
)

process.source = cms.Source("EmptySource")

process.p = cms.Path(process.Thing*process.OtherThing)
process.ep = cms.EndPath(process.output)

process.options = cms.untracked.PSet(multiProcesses=cms.untracked.PSet(
        maxChildProcesses=cms.untracked.int32(3),
        maxSequentialEventsPerChild=cms.untracked.uint32(2)))
