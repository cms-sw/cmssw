import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTTRANSIENT")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.TransientThing = cms.EDProducer("TransientIntProducer",
  ivalue = cms.int32(1)
)

process.ThingFromTransient = cms.EDProducer("IntProducerFromTransient",
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:PoolTransientTest.root')
)

process.source = cms.Source("EmptySource")

process.p = cms.Path(process.TransientThing*process.ThingFromTransient)
process.ep = cms.EndPath(process.output)


