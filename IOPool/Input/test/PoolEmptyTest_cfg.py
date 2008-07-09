# The following comments couldn't be translated into the new config version:

# Configuration file for PoolInputTest

import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITEEMPTY")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.Thing = cms.EDProducer("ThingProducer",
    debugLevel = cms.untracked.int32(1)
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('PoolEmptyTest.root')
)

process.source = cms.Source("TestRunLumiSource",
    runLumiEvent = cms.untracked.vint32(0, 0, 0)
)

process.p = cms.Path(process.Thing)
process.ep = cms.EndPath(process.output)


