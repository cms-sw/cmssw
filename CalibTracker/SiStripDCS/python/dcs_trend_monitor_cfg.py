import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("plot")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('*')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.load('Configuration.Geometry.GeometryExtended2018_cff')

process.load("CondCore.CondDB.CondDB_cfi")
process.tkVoltageTrend = cms.EDAnalyzer( "SiStripDetVOffTrendPlotter",
                                     process.CondDB,
                                     conditionDatabase = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
#                                     conditionDatabase = cms.string("oracle://cms_orcoff_prep/CMS_CONDITIONS"),
                                     # Add the tags for plotting
                                     plotTags = cms.vstring("SiStripDetVOff_1hourDelay_v1_Validation", "SiStripDetVOff_13hourDelay_v1_Validation", "SiStripDetVOff_v7_prompt"),
                                     # Set the time interval for the plots, e.g., put 48 if you want to plot the trend in the last 48 hours.
                                     # Set timeInterval to non-positive values if you want to put start and end time by hand.
                                     timeInterval = cms.int32(72),
                                     # Start and end time for plotting. Only used if timeInterval is non-positive.
                                     # Time format: "2002-01-20 23:59:59.000" (UTC).
                                     startTime = cms.untracked.string("2016-01-01 00:00:00.000"),
                                     endTime   = cms.untracked.string("2016-01-02 00:00:00.000"),
                                     # Set the name of the output plot files. Will use the timestamps if left empty.
                                     outputPlot = cms.untracked.string("last.png"),
                                     # Set output root file name. Leave empty if do not want to save plots in a root file.
                                     outputRootFile = cms.untracked.string(""),
                                     # Set output CSV file name. Leave empty if do not want to dump HV/LV counts in a CSV file.
                                     outputCSV = cms.untracked.string("last.csv")
                                     )

process.p = cms.Path(process.tkVoltageTrend)
