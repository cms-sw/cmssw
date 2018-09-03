import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("summary")

process.MessageLogger = cms.Service( "MessageLogger",
                                     debugModules = cms.untracked.vstring( "*" ),
                                     cout = cms.untracked.PSet( threshold = cms.untracked.string( "DEBUG" ) ),
                                     destinations = cms.untracked.vstring( "cout" )
                                     )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.load("CondCore.CondDB.CondDB_cfi")
process.DetVOffSummary = cms.EDAnalyzer( "SiStripDetVOffPrinter",
                                     process.CondDB,
                                     conditionDatabase = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
                                     # Add the tag 
                                     tagName = cms.string("SiStripDetVOff_1hourDelay_v1_Validation"),
                                     # Start and end time
                                     # Time format: "2002-01-20 23:59:59.000" (UTC).
                                     startTime = cms.untracked.string("2018.08.09 18:20:00"),
                                     endTime   = cms.untracked.string("2018.08.09 22:14:00"),
                                     # Set output file name. Leave empty if do not want to dump HV/LV counts in a text file.
                                     output = cms.untracked.string("PerModuleSummary.txt")
                                     )

process.p = cms.Path(process.DetVOffSummary)
