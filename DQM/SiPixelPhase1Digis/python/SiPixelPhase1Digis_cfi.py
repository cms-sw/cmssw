import FWCore.ParameterSet.Config as cms

# this might also go into te Common config,as we do not reference it
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1DigisConf = cms.VPSet(
  DefaultHisto.clone(
    enabled = True, # ADC
    name = "adc",
    title = "Digi ADC values",
    xlabel = "adc readout",
    range_min = 0,
    range_max = 300,
    range_nbins = 300,
    specs = cms.VPSet(
      Specification().groupBy(DefaultHisto.defaultGrouping) # per-ladder and profiles
                     .save()
                     .reduce("MEAN")
                     .groupBy(parent(DefaultHisto.defaultGrouping), "EXTEND_X")
                     .saveAll(),
      Specification().groupBy(parent(DefaultHisto.defaultGrouping)) # per-layer
                     .save()
    )
  ),
  DefaultHisto.clone(
    enabled = True, # Ndigis
    name = "digis", # 'Count of' added automatically
    title = "Digis",
    xlabel = "digis",
    range_min = 0,
    range_max = 30,
    range_nbins = 30,
    dimensions = 0, # this is a count
    specs = cms.VPSet(
      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/DetId/Event") 
                     .reduce("COUNT") # per-event counting
                     .groupBy(DefaultHisto.defaultGrouping) # per-ladder and profiles
                     .save()
                     .reduce("MEAN")
                     .groupBy(parent(DefaultHisto.defaultGrouping), "EXTEND_X")
                     .saveAll(),
      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/DetId/Event")
                     .reduce("COUNT")
                     .groupBy(parent(DefaultHisto.defaultGrouping)) # per-layer
                     .save(),
      Specification().groupBy("PXBarrel|PXEndcap/FED/Event")
                     .reduce("COUNT")
                     .groupBy("PXBarrel|PXEndcap/FED")
                     .groupBy("PXBarrel|PXEndcap", "EXTEND_Y")
                     .save(),
      Specification().groupBy("PXBarrel|PXEndcap/PXLayer|PXDisk/FED/Event")
                     .reduce("COUNT")
                     .groupBy("PXBarrel|PXEndcap/PXLayer|PXDisk")
                     .save()
    )
  ),
  DefaultHisto.clone(
    enabled = True, # Event Rate
    bookUndefined = True, # for now needed, since BX and Lumi are not defined in booking.
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
  ),
  DefaultHisto.clone(
    enabled = True, # hitmaps
    name = "hitmap",
    title = "Position of digis on module",
    ylabel = "#digis",
    dimensions = 0,
    specs = cms.VPSet(
      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXFModule/row/col")
                     .groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXFModule/row", "EXTEND_X")
                     .groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXFModule", "EXTEND_Y")
                     .save(),
      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXFModule/col")
                     .groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXFModule", "EXTEND_X")
                     .save(),
      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXFModule/row")
                     .groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXFModule", "EXTEND_X")
                     .save()
    )
  )
)

# TODO: names?
SiPixelPhase1DigisAnalyzer = cms.EDAnalyzer("SiPixelPhase1Digis",
        src = cms.InputTag("simSiPixelDigis"), 
        histograms = SiPixelPhase1DigisConf,
        geometry = SiPixelPhase1Geometry
)
# TODO: better clone() here instead?
SiPixelPhase1DigisHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1DigisConf,
        geometry = SiPixelPhase1Geometry
)
