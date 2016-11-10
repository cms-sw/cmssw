import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *


SiPixelPhase1RawDataNErrors = DefaultHisto.clone(
  name = "errors",
  title = "Errors",
  xlabel = "errors",
  range_min = 0, range_max = 30, range_nbins = 30,
  dimensions = 0,
  specs = cms.VPSet(
    Specification().groupBy("FED/Event")
                   .reduce("COUNT")
                   .groupBy("FED").save(),
    Specification().groupBy("FED/FEDChannel")
                   .groupBy("FED", "EXTEND_X")
                   .save()
                   .groupBy("", "EXTEND_Y")
                   .save()
  )
)

SiPixelPhase1RawDataFIFOFull = DefaultHisto.clone(
  name = "fifofull",
  title = "Type of FIFO full",
  xlabel = "FIFO (data bit #)",
  range_min = -0.5, range_max = 7.5, range_nbins = 8,
  dimensions = 1,
  specs = cms.VPSet(
    Specification().groupBy("FED").save(),
  )
)

SiPixelPhase1RawDataTBMMessage = DefaultHisto.clone(
  name = "tbmmessage",
  title = "TBM trailer message",
  xlabel = "TBM message (data bit #)",
  range_min = -0.5, range_max = 7.5, range_nbins = 8,
  dimensions = 1,
  specs = cms.VPSet(
    Specification().groupBy("FED").save(),
  )
)

SiPixelPhase1RawDataTBMType = DefaultHisto.clone(
  name = "tbmtype",
  title = "Type of TBM trailer",
  xlabel = "TBM type",
  range_min = -0.5, range_max = 4.5, range_nbins = 4,
  dimensions = 1,
  specs = cms.VPSet(
    Specification().groupBy("FED").save(),
  )
)

SiPixelPhase1RawDataTypeNErrors = DefaultHisto.clone(
  name = "nerrors_per_type",
  title = "Number of Errors per Type",
  xlabel = "Error Type",
  range_min = 0, range_max = 50, range_nbins = 51,#TODO: proper range here
  dimensions = 1,
  specs = cms.VPSet(
    Specification().groupBy("FED").save()
                   .groupBy("", "EXTEND_Y").save(),
  )
)

SiPixelPhase1RawDataConf = cms.VPSet(
  SiPixelPhase1RawDataNErrors,
  SiPixelPhase1RawDataFIFOFull,
  SiPixelPhase1RawDataTBMMessage,
  SiPixelPhase1RawDataTBMType,
  SiPixelPhase1RawDataTypeNErrors,
)

SiPixelPhase1RawDataAnalyzer = cms.EDAnalyzer("SiPixelPhase1RawData",
        src = cms.InputTag("siPixelDigis"),
        histograms = SiPixelPhase1RawDataConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1RawDataHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1RawDataConf,
        geometry = SiPixelPhase1Geometry
)
