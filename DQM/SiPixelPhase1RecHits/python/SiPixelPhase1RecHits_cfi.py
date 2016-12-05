import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1RecHitsNRecHits = DefaultHistoTrack.clone(
  name = "rechits",
  title = "RecHits",
  range_min = 0, range_max = 10, range_nbins = 10,
  xlabel = "rechits",
  dimensions = 0,
  specs = VPSet(
    StandardSpecificationTrend_Num,
    StandardSpecification2DProfile_Num
  )
)

SiPixelPhase1RecHitsClustX = DefaultHistoTrack.clone(
  name = "rechitsize_x",
  title = "X size of RecHit clusters",
  range_min = 0, range_max = 10, range_nbins = 10,
  xlabel = "RecHit X-Size",
  dimensions = 1,
  specs = VPSet(
    StandardSpecification2DProfile
  )
)

SiPixelPhase1RecHitsClustY = SiPixelPhase1RecHitsClustX.clone(
  name = "rechitsize_y",
  title = "Y size of RecHit clusters",
  xlabel = "RecHit Y-Size"
)

SiPixelPhase1RecHitsErrorX = DefaultHistoTrack.clone(
  name = "rechiterror_x",
  title = "RecHit Error in X-direction",
  range_min = 0, range_max = 0.02, range_nbins = 100,
  xlabel = "X error",
  dimensions = 1,
  specs = VPSet(
    StandardSpecification2DProfile
  )
)

SiPixelPhase1RecHitsErrorY = SiPixelPhase1RecHitsErrorX.clone(
  name = "rechiterror_y",
  title = "RecHit Error in Y-direction",
  xlabel = "Y error"
)

SiPixelPhase1RecHitsPosition = DefaultHistoTrack.clone(
  enabled = False,
  name = "rechit_pos",
  title = "Position of RecHits on Module",
  range_min   = -1, range_max   = 1, range_nbins   = 100,
  range_y_min = -4, range_y_max = 4, range_y_nbins = 100,
  xlabel = "x offset",
  ylabel = "y offset",
  dimensions = 2,
  specs = VPSet(
    Specification(PerModule).groupBy("PXBarrel/PXLayer/DetId").save(),
    Specification(PerModule).groupBy("PXForward/PXDisk/DetId").save(),
  )
)

SiPixelPhase1RecHitsProb = DefaultHistoTrack.clone(
  name = "clusterprob",
  title = "Cluster Probability",
  xlabel = "log_10(Pr)",
  range_min = -10, range_max = 1, range_nbins = 50,
  dimensions = 1,
  specs = VPSet(
    StandardSpecifications1D
  )
)


SiPixelPhase1RecHitsConf = cms.VPSet(
  SiPixelPhase1RecHitsNRecHits,
  SiPixelPhase1RecHitsClustX,
  SiPixelPhase1RecHitsClustY,
  SiPixelPhase1RecHitsErrorX,
  SiPixelPhase1RecHitsErrorY,
  SiPixelPhase1RecHitsPosition,
  SiPixelPhase1RecHitsProb,
)

SiPixelPhase1RecHitsAnalyzer = cms.EDAnalyzer("SiPixelPhase1RecHits",
        src = cms.InputTag("siPixelRecHits"),
        histograms = SiPixelPhase1RecHitsConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1RecHitsHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1RecHitsConf,
        geometry = SiPixelPhase1Geometry
)
