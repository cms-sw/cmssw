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
                     .save()
    )
  ),
  DefaultHisto.clone(
    enabled = True,  # Ndigis per FED
    name = "digis",  # This is the same as above up to the ranges. maybe we 
    title = "Digis", # should allow setting the range per spec, but OTOH a 
    xlabel = "digis",# HistogramManager is almost free.
    range_min = 0,
    range_max = 200,
    range_nbins = 200,
    dimensions = 0, 
    specs = cms.VPSet(
      Specification().groupBy("PXBarrel|PXForward/FED/Event")
                     .reduce("COUNT")
                     .groupBy("PXBarrel|PXForward/FED")
                     .groupBy("PXBarrel|PXForward", "EXTEND_Y")
                     .save(),
      Specification().groupBy("PXBarrel|PXForward/PXLayer|PXDisk/FED/Event")
                     .reduce("COUNT")
                     .groupBy("PXBarrel|PXForward/PXLayer|PXDisk")
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
    bookUndefined = True,
    name = "hitmap",
    title = "Position of digis on module",
    ylabel = "#digis",
    dimensions = 0,
    specs = cms.VPSet(
      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXPanel/row/col")
                     .groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXPanel/row", "EXTEND_X")
                     .groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXPanel", "EXTEND_Y")
                     .save()
		     .groupBy(DefaultHisto.defaultGrouping, "SUM").saveAll(),

      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXPanel/col")
                     .groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXPanel", "EXTEND_X")
                     .save()
		     .groupBy(DefaultHisto.defaultGrouping, "SUM").saveAll(),

      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXPanel/row")
                     .groupBy(DefaultHisto.defaultGrouping.value() + "/PXBModule|PXPanel", "EXTEND_X")
                     .save()
		     .groupBy(DefaultHisto.defaultGrouping, "SUM").saveAll(),

      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/ROCinLadder|ROCinBlade/LumiDecade")
                     .groupBy(DefaultHisto.defaultGrouping.value() + "/ROCinLadder|ROCinBlade", "EXTEND_X")
                     .groupBy(DefaultHisto.defaultGrouping, "EXTEND_Y")
                     .save(),

      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/ROC")
                     .groupBy(DefaultHisto.defaultGrouping,"EXTEND_X")
                     .save()
		     .groupBy(DefaultHisto.defaultGrouping, "SUM").saveAll(),

      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/ROCinLadder|ROCinBlade")
                     .groupBy(DefaultHisto.defaultGrouping,"EXTEND_X")
                     .save()
		     .groupBy(DefaultHisto.defaultGrouping, "SUM").saveAll()
    )
  ),
  DefaultHisto.clone(
    enabled = True, # Geometry Debug
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
      Specification().groupBy(parent(DefaultHisto.defaultGrouping)) # per-layer
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
