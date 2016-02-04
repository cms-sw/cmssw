# Configuration file for testing files with differing product ids   

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

process.filterModule = cms.EDFilter("TestFilterModule")

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep edmtestThings_*_*_*', 
        'keep *_TriggerResults_*_*'),
    fileName = cms.untracked.string('prod1.root')
)

process.p = cms.Path(process.Thing*process.OtherThing*process.filterModule)

# These are here for the TriggerNames test, they do not
# do anything other than add a couple more path names
process.p1 = cms.Path(process.Thing*process.OtherThing)
process.p2 = cms.Path(process.Thing*process.OtherThing)

process.outp = cms.EndPath(process.out)
