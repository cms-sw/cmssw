import FWCore.ParameterSet.Config as cms

process = cms.Process("TRANSFER")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("DQMStreamerReader",
                            SelectEvents = cms.untracked.vstring("*"),
                            runNumber =cms.untracked.uint32(1),
                            delayMillis =cms.untracked.uint32(100),
                            streamLabel = cms.untracked.string("testAlt"),
                            runInputDir = cms.untracked.string("."),
                            datafnPosition = cms.untracked.uint32(2)
    #firstEvent = cms.untracked.uint64(10123456835)
)

process.json = cms.EDAnalyzer("DQMStreamerWriteJsonAnalyzer",
                              eventsPerLumi = cms.untracked.uint32(50),
                              runNumber = cms.untracked.uint32(1),
                              streamName = cms.untracked.string("testAlt"),
                              dataFileForEachLumi = cms.untracked.vstring("teststreamfile_alt.dat"),
                              pathToWriteJson = cms.untracked.string(".")
)

process.a1 = cms.EDAnalyzer("StreamThingAnalyzer",
    product_to_get = cms.string('m1')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myout.root')
)

process.end = cms.EndPath(process.a1*process.out*process.json)
