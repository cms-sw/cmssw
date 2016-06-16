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

process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.load("CondCore.CondDB.CondDB_cfi")
process.tkVoltageTrend = cms.EDAnalyzer( "SiStripDetVOffTrendPlotter",
                                     process.CondDB,
#                                      conditionDatabase = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
                                     conditionDatabase = cms.string("oracle://cms_orcoff_prep/CMS_CONDITIONS"),
                                     # Add the tags for plotting
                                     plotTags = cms.vstring("SiStripDetVOff_test_1hr_prompt", "SiStripDetVOff_test_13hr_prompt", "SiStripDetVOff_test_25hr_prompt"),
                                     # Set the time interval for the plots, e.g., put 48 if you want to plot the trend in the last 48 hours.
                                     # Set timeInterval to non-positive values if you want to put start and end time by hand.
                                     timeInterval = cms.int32(0),
                                     # Start and end time for plotting. Only used if timeInterval is non-positive.
                                     # Time format: "2002-01-20 23:59:59.000" (UTC).
                                     startTime = cms.untracked.string("2016-03-11 16:30:50.000"),
                                     endTime   = cms.untracked.string("2016-03-20 00:00:00.000"),
                                     # Set the name of the output root file. Leave empty if do not want to save plots in a root file.
                                     outputFile = cms.untracked.string("output.root")
                                     )

process.p = cms.Path(process.tkVoltageTrend)
