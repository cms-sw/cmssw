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
      Specification().groupBy(DefaultHisto.defaultGrouping)
                     .saveAll(),
      Specification().groupBy("BX")
                     .reduce("COUNT")
                     .groupBy("", "EXTEND_X")
                     .save(),
      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/row/col")
                     .reduce("COUNT")
                     .groupBy(DefaultHisto.defaultGrouping.value() + "/row", "EXTEND_X")
                     .groupBy(DefaultHisto.defaultGrouping, "EXTEND_Y")
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
                     .reduce("COUNT")
                     .groupBy(DefaultHisto.defaultGrouping)
                     .save()
                     .reduce("MEAN")
                     .groupBy("PXBarrel|PXEndcap/PXLayer|PXDisk", "EXTEND_X")
                     .saveAll()
    )
  ),
  DefaultHisto.clone(
    enabled = True, # hitmaps
    name = "hitmap",
    title = "Position of digis on module",
    xlabel = "col",
    ylabel = "row",
    range_min = 0,
    range_max = 200,
    range_nbins = 200,
    dimensions = 2,
    specs = cms.VPSet(
      Specification().groupBy("PXBarrel|PXEndcap/PXLayer|PXDisk/PXLadder|PXBlade/DetId")
                     .save()
                     .groupBy("PXBarrel|PXEndcap/PXLayer|PXDisk/PXLadder|PXBlade", "EXTEND_X")
                     .save()
                     .groupBy("PXBarrel|PXEndcap/PXLayer|PXDisk", "SUM")
                     .saveAll()
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
