import FWCore.ParameterSet.Config as cms

# this might also go into te Common config,as we do not reference it
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1DigisADC = DefaultHistoDigiCluster.clone(
  name = "adc",
  title = "Digi ADC values",
  xlabel = "adc readout",
  range_min = 0,
  range_max = 300,
  range_nbins = 300,
  specs = VPSet(
    StandardSpecificationTrend,
    StandardSpecificationTrend2D,
    StandardSpecification2DProfile,
    StandardSpecifications1D
  )
)

SiPixelPhase1DigisNdigis = DefaultHistoDigiCluster.clone(
  name = "digis", # 'Count of' added automatically
  title = "Digis",
  xlabel = "digis",
  range_min = 0,
  range_max = 30,
  range_nbins = 30,
  dimensions = 0, # this is a count
  specs = VPSet(
    StandardSpecificationTrend_Num,
    StandardSpecification2DProfile_Num,
    StandardSpecifications1D_Num
  )
)


SiPixelPhase1ClustersNdigisInclusive = DefaultHistoDigiCluster.clone(
  name = "digis",
  title = "Digis",
  range_min = 0, range_max = 2000, range_nbins = 200,
  xlabel = "digis",
  dimensions = 0,
  specs = VPSet(
    StandardSpecificationInclusive_Num
  )
)


SiPixelPhase1DigisNdigisPerFED = DefaultHisto.clone( #to be removed?
  name = "feddigis", # This is the same as above up to the ranges. maybe we 
  title = "Digis",   # should allow setting the range per spec, but OTOH a 
  xlabel = "digis",  # HistogramManager is almost free.
  range_min = 0,
  range_max = 1000,
  range_nbins = 200,
  dimensions = 0, 
  specs = VPSet(
    Specification().groupBy("FED/Event")
                   .reduce("COUNT")
                   .groupBy("FED")
                   .groupBy("", "EXTEND_Y")
                   .save()
  )
)

SiPixelPhase1DigisNdigisPerFEDtrend = DefaultHisto.clone(                                                                                                                                                   
  name = "feddigistrend", # This is the same as above up to the ranges. maybe we                                                                                                                                            
  title = "Digis",   # should allow setting the range per spec, but OTOH a                                                                                                                                             
  xlabel = "digis",  # HistogramManager is almost free.                                                                                                                                                                
  range_min = 0,
  range_max = 1000,
  range_nbins = 200,
  dimensions = 0,
  specs = VPSet(
  Specification().groupBy("Lumisection/FED/FED/Event")
                   .reduce("COUNT")
                   .groupBy("Lumisection/FED")
                   .reduce("MEAN")
                   .groupBy("Lumisection", "EXTEND_Y")
                   .groupBy("", "EXTEND_X")
                   .save()
  )
)

SiPixelPhase1DigisEvents = DefaultHistoDigiCluster.clone(
  name = "eventrate",
  title = "Rate of Pixel Events",
  xlabel = "Lumisection",
  ylabel = "#Events",
  dimensions = 0,
  specs = VPSet(
    Specification().groupBy("Lumisection")
                   .groupBy("", "EXTEND_X").save(),
    Specification().groupBy("BX")
                   .groupBy("", "EXTEND_X").save()
  )
)

SiPixelPhase1DigisHitmap = DefaultHistoDigiCluster.clone(
  name = "hitmap",
  title = "Position of digis on module",
  ylabel = "#digis",
  dimensions = 0,
  specs = VPSet(
    Specification(PerModule).groupBy("PXBarrel/Shell/PXLayer/PXLadder/PXModuleName/row/col")
                   .groupBy("PXBarrel/Shell/PXLayer/PXLadder/PXModuleName/row", "EXTEND_Y")
                   .groupBy("PXBarrel/Shell/PXLayer/PXLadder/PXModuleName", "EXTEND_X")
                   .save(),
    Specification(PerModule).groupBy("PXBarrel/Shell/PXLayer/PXLadder/PXModuleName/col")
                   .groupBy("PXBarrel/Shell/PXLayer/PXLadder/PXModuleName", "EXTEND_X")
                   .save(),
    Specification(PerModule).groupBy("PXBarrel/Shell/PXLayer/PXLadder/PXModuleName/row")
                   .groupBy("PXBarrel/Shell/PXLayer/PXLadder/PXModuleName", "EXTEND_X")
                   .save(),

    Specification(PerModule).groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade/PXModuleName/row/col")
                   .groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade/PXModuleName/row", "EXTEND_Y")
                   .groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade/PXModuleName", "EXTEND_X")
                   .save(),
    Specification(PerModule).groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade/PXModuleName/col")
                   .groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade/PXModuleName", "EXTEND_X")
                   .save(),
    Specification(PerModule).groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade/PXModuleName/row")
                   .groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade/PXModuleName", "EXTEND_X")
                   .save(),
    StandardSpecificationOccupancy,
  )
)

SiPixelPhase1DigisOccupancy = DefaultHistoReadout.clone(
  name = "occupancy",
  title = "Digi Occupancy",
  dimensions = 0,
  specs = VPSet(
    Specification(PerReadout).groupBy("PXBarrel/FED/LinkInFed")
                             .groupBy("PXBarrel/FED", "EXTEND_X").save(),
    Specification(PerReadout).groupBy("PXBarrel/FED/LinkInFed/RocInLink")
                             .groupBy("PXBarrel/FED/LinkInFed", "EXTEND_Y")
                             .groupBy("PXBarrel/FED", "EXTEND_X").save(),
    Specification(PerReadout).groupBy("PXForward/FED/LinkInFed")
                             .groupBy("PXForward/FED", "EXTEND_X").save(),
    Specification(PerReadout).groupBy("PXForward/FED/LinkInFed/RocInLink")
                             .groupBy("PXForward/FED/LinkInFed", "EXTEND_Y")
                             .groupBy("PXForward/FED", "EXTEND_X").save(),
    Specification(PerReadout).groupBy("PXBarrel/FED")
                             .groupBy("PXBarrel", "EXTEND_X").save(),
    Specification(PerReadout).groupBy("PXForward/FED")
                             .groupBy("PXForward", "EXTEND_X").save(),

  )
)

# This has to match the order of the names in the C++ enum.
SiPixelPhase1DigisConf = cms.VPSet(
  SiPixelPhase1DigisADC,
  SiPixelPhase1DigisNdigis,
  SiPixelPhase1ClustersNdigisInclusive,
  SiPixelPhase1DigisNdigisPerFED,
  SiPixelPhase1DigisNdigisPerFEDtrend,
  SiPixelPhase1DigisEvents,
  SiPixelPhase1DigisHitmap,
  SiPixelPhase1DigisOccupancy,
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
