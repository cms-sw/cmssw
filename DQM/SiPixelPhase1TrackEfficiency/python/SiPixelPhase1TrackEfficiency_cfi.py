import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1TrackEfficiencyValid = DefaultHistoTrack.clone(
  name = "valid",
  title = "Valid Hits",
  xlabel = "valid hits",
  dimensions = 0,
  specs = VPSet(
    StandardSpecifications1D_Num,
    StandardSpecificationOccupancy,
  )
)

SiPixelPhase1TrackEfficiencyMissing = DefaultHistoTrack.clone(
  name = "missing",
  title = "Missing Hits",
  xlabel = "missing hits",
  dimensions = 0,
  specs = VPSet(
    StandardSpecifications1D_Num,
    StandardSpecificationOccupancy,
  )
)

SiPixelPhase1TrackEfficiencyEfficiency = SiPixelPhase1TrackEfficiencyValid.clone(
  name = "hitefficiency",
  title = "Hit Efficiency",
  xlabel = "#valid/(#valid+#missing)",
  dimensions = 1,
  specs = VPSet(
    StandardSpecification2DProfile,
    StandardSpecificationPixelmapProfile,
    Specification().groupBy("PXBarrel/PXLayer/ROC")
                   .groupBy("PXBarrel/PXLayer", "EXTEND_X")
                   .reduce("MEAN")
                   .save(),
    Specification().groupBy("PXForward/PXRing/ROC")
                   .groupBy("PXForward/PXRing", "EXTEND_X")
                   .reduce("MEAN")
                   .save()
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
        trajectories = cms.InputTag("generalTracks"),
        primaryvertices = cms.InputTag("offlinePrimaryVertices"),
        histograms = SiPixelPhase1TrackEfficiencyConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1TrackEfficiencyHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1TrackEfficiencyConf,
        geometry = SiPixelPhase1Geometry
)
