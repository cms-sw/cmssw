# The following comments couldn't be translated into the new config version:

# Configuration file for PrePoolInputTest 

import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTPROD")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(11)
)
process.Thing = cms.EDProducer("ThingProducer")

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *", "drop *_Thing_*_TESTPROD"),
    fileName = cms.untracked.string('aliastest_step1_diffOrder.root')
)

process.source = cms.Source("EmptySource",
    firstLuminosityBlock = cms.untracked.uint32(6),
    numberEventsInLuminosityBlock = cms.untracked.uint32(3),
    firstRun = cms.untracked.uint32(561),
    numberEventsInRun = cms.untracked.uint32(7)
)

process.OtherThing = cms.EDProducer("OtherThingProducer",
    thingTag = cms.InputTag("ZThing")
    )

#This changes the order of the original and alias branch names and branchIDs
process.ZThing = cms.EDAlias(
    Thing = cms.VPSet(
      cms.PSet(type = cms.string('edmtestThings'),
               fromProductInstance = cms.string('*'),
               toProductInstance = cms.string('*'))
    )
)


process.p = cms.Path(process.Thing*process.OtherThing)
process.ep = cms.EndPath(process.output)

