# Configuration file for RefTest_t   

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

process.OtherThing = cms.EDProducer("OtherThingProducer",
    debugLevel = cms.untracked.int32(1)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('good.root')
)

process.out2 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('good2.root')
)

process.out_other = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep edmtestOtherThings_*_*_*', 
        'keep *_TriggerResults_*_*'),
    fileName = cms.untracked.string('other_only.root')
)


process.p = cms.Path(process.Thing*process.OtherThing)
process.outp = cms.EndPath(process.out*process.out2*process.out_other)
