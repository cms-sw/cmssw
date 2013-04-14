# Configuration file for RefTest_t   

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource")

process.Thing = cms.EDProducer("ThingProducer",
    offsetDelta = cms.int32(5)
)

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('good_b.root')
)

process.p = cms.Path(process.Thing*process.OtherThing)
process.outp = cms.EndPath(process.out)
