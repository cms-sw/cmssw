import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *


SiPixelPhase1TrackEfficiencyValid = DefaultHistoTrack.clone(
  name = "valid",
  title = "Valid Hits",
  range_min = 0, range_max = 50, range_nbins = 50,
  xlabel = "valid hits",
  dimensions = 0,

  specs = VPSet(
    StandardSpecifications1D_Num,
    #StandardSpecification2DProfile_Num, #for this we have the on track clusters map (i.e the same thing)

    Specification().groupBy("PXBarrel/PXLayer/Event") #this will produce inclusive counts per Layer/Disk
                             .reduce("COUNT")    
                             .groupBy("PXBarrel/PXLayer")
                             .save(nbins=50, xmin=0, xmax=1500),
    Specification().groupBy("PXForward/PXDisk/Event")
                             .reduce("COUNT")    
                             .groupBy("PXForward/PXDisk/")
                             .save(nbins=50, xmin=0, xmax=1500),
  )
)

SiPixelPhase1TrackEfficiencyInactive = DefaultHistoTrack.clone(
  name = "inactive",
  title = "Inactive Hits",
  xlabel = "inactive hits",
  range_min = 0, range_max = 25, range_nbins = 25,
  dimensions = 0,

  specs = VPSet(
    StandardSpecification2DProfile_Num,

    Specification().groupBy("PXBarrel/PXLayer/Event") #this will produce inclusive counts per Layer/Disk
                             .reduce("COUNT")    
                             .groupBy("PXBarrel/PXLayer")
                             .save(nbins=50, xmin=0, xmax=100),
    Specification().groupBy("PXForward/PXDisk/Event")
                             .reduce("COUNT")    
                             .groupBy("PXForward/PXDisk/")
                             .save(nbins=50, xmin=0, xmax=100),
  )
)

SiPixelPhase1TrackEfficiencyMissing = DefaultHistoTrack.clone(
  name = "missing",
  title = "Missing Hits",
  range_min = 0, range_max = 25, range_nbins = 25,
  xlabel = "missing hits",
  dimensions = 0,

  specs = VPSet(
    StandardSpecifications1D_Num,
    StandardSpecification2DProfile_Num,

    Specification().groupBy("PXBarrel/PXLayer/Event") #this will produce inclusive counts per Layer/Disk
                             .reduce("COUNT")    
                             .groupBy("PXBarrel/PXLayer")
                             .save(nbins=50, xmin=0, xmax=100),
    Specification().groupBy("PXForward/PXDisk/Event")
                             .reduce("COUNT")    
                             .groupBy("PXForward/PXDisk/")
                             .save(nbins=50, xmin=0, xmax=100),
  )
)

SiPixelPhase1TrackEfficiencyEfficiency = SiPixelPhase1TrackEfficiencyValid.clone(
  name = "hitefficiency",
  title = "Hit Efficiency",
  xlabel = "#valid/(#valid+#missing)",
  dimensions = 1,
  specs = VPSet(
    #2D profile maps per layer
    StandardSpecification2DProfile,

    #profiles per layer and shell
    Specification(PerLadder).groupBy("PXBarrel/Shell/PXLayer/SignedLadder")
                            .reduce("MEAN")
                            .groupBy("PXBarrel/Shell/PXLayer", "EXTEND_X")
                            .save(),
    Specification(PerLadder).groupBy("PXForward/HalfCylinder/PXRing/PXDisk/SignedBlade")
                            .reduce("MEAN")
                            .groupBy("PXForward/HalfCylinder/PXRing/PXDisk", "EXTEND_X")
                            .save(),
    #per layer
    Specification().groupBy("PXBarrel/PXLayer")
                   .reduce("MEAN")
                   .groupBy("PXBarrel", "EXTEND_X")
                   .save(),
    Specification().groupBy("PXForward/PXDisk")
                   .reduce("MEAN")
                   .groupBy("PXForward", "EXTEND_X")
                   .save()

    #StandardSpecificationPixelmapProfile    
  )
)

SiPixelPhase1TrackEfficiencyVertices= DefaultHistoTrack.clone(
    name = "num_vertices",
    title = "PrimaryVertices",
    xlabel= "# Vertices",
    dimensions = 1,
    range_min = -0.5,
    range_max = 100.5, 
    range_nbins =101,
    specs = VPSet(
        Specification().groupBy("")
                   .save(),
        Specification().groupBy("/Lumisection")
                   .reduce("MEAN")
                   .groupBy("","EXTEND_X")
                   .save()
   )
)



SiPixelPhase1TrackEfficiencyConf = cms.VPSet(
  SiPixelPhase1TrackEfficiencyValid,
  SiPixelPhase1TrackEfficiencyMissing,
  SiPixelPhase1TrackEfficiencyInactive,
  SiPixelPhase1TrackEfficiencyEfficiency,
  SiPixelPhase1TrackEfficiencyVertices
)


SiPixelPhase1TrackEfficiencyAnalyzer = cms.EDAnalyzer("SiPixelPhase1TrackEfficiency",
        clusters = cms.InputTag("siPixelClusters"),
        tracks = cms.InputTag("generalTracks"),
        primaryvertices = cms.InputTag("offlinePrimaryVertices"),
        histograms = SiPixelPhase1TrackEfficiencyConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1TrackEfficiencyHarvester = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1TrackEfficiencyConf,
        geometry = SiPixelPhase1Geometry
)
