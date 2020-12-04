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

process.load("DQM.SiStripCommon.TkHistoMap_cff")
# load TrackerTopology (needed for TkDetMap and TkHistoMap)
process.load("Configuration.Geometry.GeometryExtended2017_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.trackerTopology = cms.ESProducer("TrackerTopologyEP")

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
