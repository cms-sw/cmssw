import FWCore.ParameterSet.Config as cms

# this might also go into te Common config,as we do not reference it
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1DigisADC = DefaultHisto.clone(
  name = "adc",
  title = "Digi ADC values",
  xlabel = "adc readout",
  range_min = 0,
  range_max = 300,
  range_nbins = 300,
  specs = cms.VPSet(
    StandardSpecificationTrend,
    StandardSpecification2DProfile,
    *StandardSpecifications1D
  )
)

SiPixelPhase1DigisNdigis = DefaultHisto.clone(
  name = "digis", # 'Count of' added automatically
  title = "Digis",
  xlabel = "digis",
  range_min = 0,
  range_max = 30,
  range_nbins = 30,
  dimensions = 0, # this is a count
  specs = cms.VPSet(
    StandardSpecificationTrend_Num,
    StandardSpecification2DProfile_Num,
    *StandardSpecifications1D_Num
  )
)

SiPixelPhase1DigisNdigisPerFED = DefaultHisto.clone(
  name = "digis",  # This is the same as above up to the ranges. maybe we 
  title = "Digis", # should allow setting the range per spec, but OTOH a 
  xlabel = "digis",# HistogramManager is almost free.
  range_min = 0,
  range_max = 1000,
  range_nbins = 200,
  dimensions = 0, 
  specs = cms.VPSet(
    Specification().groupBy("FED/Event")
                   .reduce("COUNT")
                   .groupBy("FED")
                   .groupBy("", "EXTEND_Y")
                   .save()
  )
)

SiPixelPhase1DigisEvents = DefaultHisto.clone(
  name = "eventrate",
  title = "Rate of Pixel Events",
  xlabel = "Lumisection",
  ylabel = "#Events",
  dimensions = 0,
  specs = cms.VPSet(
    Specification().groupBy("Lumisection")
                   .groupBy("", "EXTEND_X").save(),
    Specification().groupBy("BX")
                   .groupBy("", "EXTEND_X").save()
  )
)

SiPixelPhase1DigisHitmap = DefaultHisto.clone(
  name = "hitmap",
  title = "Position of digis on module",
  ylabel = "#digis",
  dimensions = 0,
  specs = cms.VPSet(
    Specification(PerModule).groupBy(DefaultHisto.defaultPerModule.value() + "/row/col")
                   .groupBy(DefaultHisto.defaultPerModule.value() + "/row", "EXTEND_X")
                   .groupBy(DefaultHisto.defaultPerModule.value(), "EXTEND_Y")
                   .save(),
    Specification(PerModule).groupBy(DefaultHisto.defaultPerModule.value() + "/col")
                   .groupBy(DefaultHisto.defaultPerModule.value(), "EXTEND_X")
                   .save(),
    Specification(PerModule).groupBy(DefaultHisto.defaultPerModule.value() + "/row")
                   .groupBy(DefaultHisto.defaultPerModule.value(), "EXTEND_X")
                   .save()

  )
)

SiPixelPhase1DigisDebug = DefaultHisto.clone(
  enabled = False,
  name = "debug",
  xlabel = "ladder #",
  range_min = 1,
  range_max = 64,
  range_nbins = 64,
  specs = cms.VPSet(
    Specification().groupBy(DefaultHisto.defaultGrouping) 
                   .save()
                   .reduce("MEAN")
                   .groupBy(parent(DefaultHisto.defaultGrouping), "EXTEND_X")
                   .saveAll(),
  )
)

# This has to match the order of the names in the C++ enum.
SiPixelPhase1DigisConf = cms.VPSet(
  SiPixelPhase1DigisADC,
  SiPixelPhase1DigisNdigis,
  SiPixelPhase1DigisNdigisPerFED,
  SiPixelPhase1DigisEvents,
  SiPixelPhase1DigisHitmap,
  SiPixelPhase1DigisDebug
)

SiPixelPhase1DigisAnalyzer = cms.EDAnalyzer("SiPixelPhase1Digis",
        src = cms.InputTag("simSiPixelDigis"), 
        histograms = SiPixelPhase1DigisConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1DigisHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1DigisConf,
        geometry = SiPixelPhase1Geometry
)
