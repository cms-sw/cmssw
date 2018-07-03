# Configuration file for PrePoolInputTest 

import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTPROD")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(11)
)
process.Thing = cms.EDProducer("ThingProducer")

process.source = cms.Source("EmptySource",
    firstLuminosityBlock = cms.untracked.uint32(6),
    numberEventsInLuminosityBlock = cms.untracked.uint32(3),
    firstRun = cms.untracked.uint32(561),
    numberEventsInRun = cms.untracked.uint32(7)
)


process.OtherThing = cms.EDProducer("OtherThingProducer",
    thingTag = cms.InputTag("AltThing"))

process.AltThing = cms.EDAlias(
    Thing = cms.VPSet(
      cms.PSet(type = cms.string('edmtestThings'),
               fromProductInstance = cms.string('*'),
               toProductInstance = cms.string('*'))
    )
)

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *", "drop *_Thing_*_TESTPROD"),
    fileName = cms.untracked.string('step1.root')
)


process.p = cms.Path(process.Thing*process.OtherThing)
process.ep = cms.EndPath(process.output)

analysisProcess = cms.Process("TESTANALYSIS")

process.addSubProcess(cms.SubProcess(analysisProcess,
   # Optional SelectEvents parameter can go here.
   #outputCommands = cms.untracked.vstring("drop *")
   outputCommands = cms.untracked.vstring("keep *", "drop *_Thing_*_TESTPROD"))
)

# Configuration file for PoolInputTest

analysisProcess.Analysis = cms.EDAnalyzer("OtherThingAnalyzer")


analysisProcess.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *", "drop *_Thing_*_TESTPROD"),
    fileName = cms.untracked.string('step2.root')
)

analysisProcess.p = cms.Path(analysisProcess.Analysis)
analysisProcess.ep = cms.EndPath(analysisProcess.output)



