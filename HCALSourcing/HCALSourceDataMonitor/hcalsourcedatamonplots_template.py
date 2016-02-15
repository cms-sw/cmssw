import FWCore.ParameterSet.Config as cms

process = cms.PSet()

process.HCALSourceDataMonPlots = cms.PSet(
    RootInputFileName = cms.string('XXX_NTUPLEFILE_XXX'),
    RootOutputFileName = cms.string('XXX_PLOTFILE_XXX'),
    NewRowEvery = cms.int32(4),
    ThumbnailSize = cms.int32(350),
    OutputRawHistograms = cms.bool(False),
    SelectDigiBasedOnTubeName = cms.bool(True),
    MaxEvents = cms.int32(2000000),
    HtmlFileName = cms.string('index.html'),
    HtmlDirName = cms.string('html'),
    PlotsDirName = cms.string('plots')

)
