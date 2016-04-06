import FWCore.ParameterSet.Config as cms

# this might also go into te Common config,as we do not reference it
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1DigisConf = cms.VPSet(
  DefaultHisto.clone(
    enabled = True, # ADC
    specs = cms.VPSet(
      Specification().groupBy(DefaultHisto.defaultGrouping)
                     .saveAll()
                     .end(),
      Specification().groupBy("BX")
                     .reduce("COUNT")
                     .groupBy("", "EXTEND_X")
                     .save()
                     .end(),
      Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/row/col")
                     .reduce("COUNT")
                     .groupBy(DefaultHisto.defaultGrouping.value() + "/row", "EXTEND_X")
                     .groupBy(DefaultHisto.defaultGrouping, "EXTEND_Y")
                     .save()
                     .end())
  ),
  DefaultHisto.clone(
    enabled = True, # Ndigis
    specs = cms.VPSet(
      Specification().groupBy(DefaultHisto.defaultGrouping)
                     .save()
                     .reduce("MEAN")
                     .groupBy("P1PXBBarrel|P1PXECEndcap/P1PXBLayer|P1PXECHalfDisk", "EXTEND_X")
                     .saveAll()
                     .end())
  ),
  DefaultHisto.clone(
    enabled = True, # hitmaps
    specs = cms.VPSet(
      Specification().groupBy("P1PXBBarrel|P1PXECEndcap/P1PXBLayer|P1PXECHalfDisk/P1PXBLadder|P1PXECBlade/DetId")
                     .save()
                     .groupBy("P1PXBBarrel|P1PXECEndcap/P1PXBLayer|P1PXECHalfDisk/P1PXBLadder|P1PXECBlade", "EXTEND_X")
                     .save()
                     .groupBy("P1PXBBarrel|P1PXECEndcap/P1PXBLayer|P1PXECHalfDisk", "SUM")
                     .saveAll()
                     .end())
  )
)

SiPixelPhase1DigisAnalyzer = cms.EDAnalyzer("SiPixelPhase1DigisAnalyzer",
        src = cms.InputTag("simSiPixelDigis"), #TODO: this should be centralized
        histograms = SiPixelPhase1DigisConf
)
SiPixelPhase1DigisHarvester = cms.EDAnalyzer("SiPixelPhase1DigisHarvester",
        histograms = SiPixelPhase1DigisConf
)
