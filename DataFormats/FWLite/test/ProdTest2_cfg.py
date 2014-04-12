# Configuration file for testing files with differing product ids   

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource",
    firstLuminosityBlock = cms.untracked.uint32(20),
    firstRun = cms.untracked.uint32(10)
)

process.Thing = cms.EDProducer("ThingProducer",
    offsetDelta = cms.int32(5)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('prod2.root')
)

process.p = cms.Path(process.Thing)
process.outp = cms.EndPath(process.out)
