# Configuration file for PartialEventTest_cfg.py   

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.Thing = cms.EDProducer("ThingProducer",
    offsetDelta = cms.untracked.int32(1),
    debugLevel = cms.untracked.int32(1)
)

process.filter = cms.EDFilter("TestFilterModule",
    onlyOne = cms.untracked.bool(True),
    acceptValue = cms.untracked.int32(2)
)

process.OtherThing = cms.EDProducer("OtherThingProducer",
    debugLevel = cms.untracked.int32(1)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('partialEvent.root')
)

process.p = cms.Path( (process.Thing+process.filter)*process.OtherThing)
process.outp = cms.EndPath(process.out)
