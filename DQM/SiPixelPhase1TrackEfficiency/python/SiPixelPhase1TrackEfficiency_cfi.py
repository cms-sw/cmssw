import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *


SiPixelPhase1TrackEfficiencyValid = DefaultHistoTrack.clone(
  name = "valid",
  title = "Valid Hits",
  xlabel = "valid hits",
  dimensions = 0,

  specs = VPSet(
    StandardSpecifications1D_Num,
    StandardSpecification2DOccupancy,

    Specification().groupBy("PXBarrel/PXLayer/Event") #this will produce inclusive counts per Layer/Disk
                             .reduce("COUNT")    
                             .groupBy("PXBarrel/PXLayer")
                             .save(nbins=100, xmin=0, xmax=3000),
    Specification().groupBy("PXForward/PXDisk/Event")
                             .reduce("COUNT")    
                             .groupBy("PXForward/PXDisk/")
                             .save(nbins=100, xmin=0, xmax=3000),
  )
)

SiPixelPhase1TrackEfficiencyMissing = DefaultHistoTrack.clone(
  name = "missing",
  title = "Missing Hits",
  xlabel = "missing hits",
  dimensions = 0,

  specs = VPSet(
    StandardSpecifications1D_Num,
    StandardSpecification2DOccupancy,

    Specification().groupBy("PXBarrel/PXLayer/Event") #this will produce inclusive counts per Layer/Disk
                             .reduce("COUNT")    
                             .groupBy("PXBarrel/PXLayer")
                             .save(nbins=100, xmin=0, xmax=100),
    Specification().groupBy("PXForward/PXDisk/Event")
                             .reduce("COUNT")    
                             .groupBy("PXForward/PXDisk/")
                             .save(nbins=100, xmin=0, xmax=100),
  )
)

SiPixelPhase1TrackEfficiencyEfficiency = SiPixelPhase1TrackEfficiencyValid.clone(
  name = "hitefficiency",
  title = "Hit Efficiency",
  xlabel = "#valid/(#valid+#missing)",
  dimensions = 1,
  specs = VPSet(
    StandardSpecification2DProfile
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
