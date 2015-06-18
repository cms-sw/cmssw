# Configuration file for RefTest_t

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.Thing = cms.EDProducer("ThingProducer")

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testTFWLiteSelector.root')
)

process.p = cms.Path(process.Thing*process.OtherThing)
process.o = cms.EndPath(process.out)
