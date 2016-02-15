import FWCore.ParameterSet.Config as cms

process = cms.PSet()

process.HCALSourceDataMonPlots = cms.PSet(
    RootInputFileName = cms.string('hcalSourceDataMon.test.215595.root'),
    RootOutputFileName = cms.string('215595_plots_hcalSourceDataMon.root'),
    NewRowEvery = cms.int32(4),
    ThumbnailSize = cms.int32(350),
    OutputRawHistograms = cms.bool(False),
    SelectDigiBasedOnTubeName = cms.bool(True),
    MaxEvents = cms.int32(2000000),
    HtmlFileName = cms.string('index.html'),
    HtmlDirName = cms.string('html'),
    PlotsDirName = cms.string('plots')

)
