import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

# this might also go into te Common config,as we do not reference it
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1DigisADC = DefaultHistoDigiCluster.clone(
  name = "adc",
  title = "Digi ADC values",
  xlabel = "adc readout",
  range_min = -0.5,
  range_max = 255.5,
  range_nbins = 32,
  specs = VPSet(
    StandardSpecificationTrend,
    StandardSpecificationTrend2D,
    StandardSpecificationPixelmapProfile,# ROC level map
    StandardSpecification2DProfile, # module level map
    StandardSpecifications1D
  )
)

SiPixelPhase1DigisNdigis = DefaultHistoDigiCluster.clone(
  name = "digis", # 'Count of' added automatically
  title = "Digis",
  xlabel = "digis",
  range_min = 0,
  range_max = 200,
  range_nbins = 100,
  dimensions = 0, # this is a count

  specs = VPSet(
    StandardSpecificationTrend_Num,
    StandardSpecification2DProfile_Num,
    StandardSpecifications1D_Num,
	
    Specification().groupBy("PXBarrel/PXLayer/Event") #this will produce inclusive counts per Layer/Disk
                             .reduce("COUNT")    
                             .groupBy("PXBarrel/PXLayer")
                             .save(nbins=100, xmin=0, xmax=20000),
    Specification().groupBy("PXForward/PXDisk/Event")
                             .reduce("COUNT")    
                             .groupBy("PXForward/PXDisk/")
                             .save(nbins=100, xmin=0, xmax=10000),
  )
)


SiPixelPhase1ClustersNdigisInclusive = DefaultHistoDigiCluster.clone(
  name = "digis",
  title = "Digis",
  range_min = 0, range_max = 100000, range_nbins = 100,
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
  range_max = 4000,
  range_nbins = 200,
  dimensions = 0, 
  specs = VPSet(
    Specification().groupBy("FED/Event")
                   .reduce("COUNT")
                   .groupBy("FED")
                   .reduce("MEAN")
                   .groupBy("", "EXTEND_X")
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
  #enabled = False,
  specs = VPSet(
  Specification().groupBy("FED/Event") #produce the mean number of digis per event and FED per lumisection
                   .reduce("COUNT")
                   .groupBy("FED/LumiBlock")
                   .reduce("MEAN")
                   .groupBy("FED", "EXTEND_X")
                   .groupBy("", "EXTEND_Y")
                   .save(),
  Specification().groupBy("FED/Event") #produce the mean number of digis per event and FED per lumisection
                   .reduce("COUNT")
                   .groupBy("LumiBlock")
                   .reduce("MEAN")
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
                   .reduce("MEAN")
                   .groupBy("", "EXTEND_X").save(),
    Specification().groupBy("BX")
                   .groupBy("", "EXTEND_X").save()
  )
)

SiPixelPhase1DigisHitmap = DefaultHistoDigiCluster.clone(
  name = "digi_occupancy",
  title = "Digi Occupancy",
  ylabel = "#digis",
  dimensions = 0,
  specs = VPSet(
    Specification(PerModule).groupBy("PXBarrel/Shell/PXLayer/SignedLadder/PXModuleName/row/col")
                   .groupBy("PXBarrel/Shell/PXLayer/SignedLadder/PXModuleName/row", "EXTEND_X")
                   .groupBy("PXBarrel/Shell/PXLayer/SignedLadder/PXModuleName", "EXTEND_Y")
                   .save(),
    Specification(PerModule).groupBy("PXBarrel/Shell/PXLayer/SignedLadder/PXModuleName/col")
                   .groupBy("PXBarrel/Shell/PXLayer/SignedLadder/PXModuleName", "EXTEND_X")
                   .save(),
    Specification(PerModule).groupBy("PXBarrel/Shell/PXLayer/SignedLadder/PXModuleName/row")
                   .groupBy("PXBarrel/Shell/PXLayer/SignedLadder/PXModuleName", "EXTEND_X")
                   .save(),
    Specification(PerModule).groupBy("PXForward/HalfCylinder/PXRing/PXDisk/SignedBlade/PXModuleName/row/col")
                   .groupBy("PXForward/HalfCylinder/PXRing/PXDisk/SignedBlade/PXModuleName/row", "EXTEND_X")
                   .groupBy("PXForward/HalfCylinder/PXRing/PXDisk/SignedBlade/PXModuleName", "EXTEND_Y")
                   .save(),
    Specification(PerModule).groupBy("PXForward/HalfCylinder/PXRing/PXDisk/SignedBlade/PXModuleName/col")
                   .groupBy("PXForward/HalfCylinder/PXRing/PXDisk/SignedBlade/PXModuleName", "EXTEND_X")
                   .save(),
    Specification(PerModule).groupBy("PXForward/HalfCylinder/PXRing/PXDisk/SignedBlade/PXModuleName/row")
                   .groupBy("PXForward/HalfCylinder/PXRing/PXDisk/SignedBlade/PXModuleName", "EXTEND_X")
                   .save(),
    StandardSpecificationOccupancy,
  )
)

SiPixelPhase1DigisOccupancy = DefaultHistoReadout.clone(
  name = "occupancy",
  title = "Digi Occupancy",
  dimensions = 0,
  specs = VPSet(
    Specification(PerReadout).groupBy("PXBarrel/FED/Channel")
                             .groupBy("PXBarrel/FED", "EXTEND_X").save(),

    #Specification(PerReadout).groupBy("PXBarrel/FED/Channel/RocInLink") #Deactivating 2D maps giving redundant information
    #                         .groupBy("PXBarrel/FED/Channel", "EXTEND_Y")
    #                         .groupBy("PXBarrel/FED", "EXTEND_X").save(),

    Specification(PerReadout).groupBy("PXForward/FED/Channel")
                             .groupBy("PXForward/FED", "EXTEND_X").save(),

    #Specification(PerReadout).groupBy("PXForward/FED/Channel/RocInLink")
    #                         .groupBy("PXForward/FED/Channel", "EXTEND_Y")
    #                         .groupBy("PXForward/FED", "EXTEND_X").save(),

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
        src = cms.InputTag("siPixelDigis"), 
        histograms = SiPixelPhase1DigisConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1DigisHarvester = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1DigisConf,
        geometry = SiPixelPhase1Geometry
)
