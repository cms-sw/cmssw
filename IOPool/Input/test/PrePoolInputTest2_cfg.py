# The following comments couldn't be translated into the new config version:

# Test storing OtherThing as well
# Configuration file for PrePoolInputTest 

import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTPROD")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(11)
)
process.Thing = cms.EDProducer("ThingProducer",
    debugLevel = cms.untracked.int32(1)
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('PoolInputTest.root')
)

process.output2 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *', 
        'drop *_Thing_*_*'),
    fileName = cms.untracked.string('PoolInputDropTest.root')
)

process.OtherThing = cms.EDProducer("OtherThingProducer",
    debugLevel = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource",
    firstLuminosityBlock = cms.untracked.uint32(6),
    numberEventsInLuminosityBlock = cms.untracked.uint32(3),
    firstRun = cms.untracked.uint32(561),
    numberEventsInRun = cms.untracked.uint32(7)
)

process.p = cms.Path(process.Thing*process.OtherThing)
process.ep = cms.EndPath(process.output*process.output2)


