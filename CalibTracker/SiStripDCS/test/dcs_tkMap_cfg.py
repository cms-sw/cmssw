import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("plot")

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

process.TkDetMap = cms.Service("TkDetMap")
process.load("DQMServices.Core.DQMStore_cfg")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.load("CondCore.CondDB.CondDB_cfi")
process.tkVoltageMap = cms.EDAnalyzer( "SiStripDetVOffTkMapPlotter",
                                     process.CondDB,
#                                      conditionDatabase = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
                                     conditionDatabase = cms.string("oracle://cms_orcoff_prep/CMS_CONDITIONS"),
                                     # Add the tag for plotting
                                     Tag = cms.string("SiStripDetVOff_test_1hr_prompt"),
                                     # Set the IOV to plot. Set to 0 if want to use a time string.
                                     IOV = cms.untracked.uint64(0),
                                     # Time format: "2002-01-20 23:59:59.000" (UTC). Setting IOV=0 and Time="" will get the last IOV.
                                     Time = cms.untracked.string("2016-03-20 00:00:00.000"),
                                     # Set the name of the output root file. Leave empty if do not want to save plots in a root file.
                                     outputFile = cms.untracked.string("tkMap.root")
                                     )

process.p = cms.Path(process.tkVoltageMap)
