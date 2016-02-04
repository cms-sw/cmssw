# Configuration file for testing files with differing product ids   

import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.Thing = cms.EDProducer("ThingProducer",
    offsetDelta = cms.untracked.int32(10),
    debugLevel = cms.untracked.int32(1)
)

process.OtherThing = cms.EDProducer("OtherThingProducer",
    debugLevel = cms.untracked.int32(1)
)

process.source = cms.Source("PoolSource",
    # hack until metadata pruning is implemented
    fileNames = cms.untracked.vstring('file:prod1.root', 
        'file:prod2.root')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('prodmerge.root')
)

process.p = cms.Path(process.Thing*process.OtherThing)
process.outp = cms.EndPath(process.out)
