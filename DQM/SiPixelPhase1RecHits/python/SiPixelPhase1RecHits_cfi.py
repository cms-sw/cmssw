import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1RecHitsNRecHits = DefaultHistoTrack.clone(
  name = "rechits",
  title = "RecHits",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "rechits",
  dimensions = 0,
  specs = VPSet(
   
   StandardSpecificationTrend_Num,
   Specification().groupBy("PXBarrel/Event")
                   .reduce("COUNT")
                   .groupBy("PXBarrel")
                   .save(nbins=100, xmin=0, xmax=5000),

    Specification().groupBy("PXForward/Event")
                   .reduce("COUNT")
                   .groupBy("PXForward")
                   .save(nbins=100, xmin=0, xmax=5000),

    Specification().groupBy("PXAll/Event")
                   .reduce("COUNT")
                   .groupBy("PXAll")
                   .save(nbins=100, xmin=0, xmax=5000)

  )
)

SiPixelPhase1RecHitsClustX = DefaultHistoTrack.clone(
  name = "clustersize_x",
  title = "Cluster Size X (OnTrack)",
  range_min = 0, range_max = 50, range_nbins = 50,
  xlabel = "size[pixels]",
  dimensions = 1,
  specs = VPSet(
    StandardSpecification2DProfile
  )
)

SiPixelPhase1RecHitsClustY = SiPixelPhase1RecHitsClustX.clone(
  name = "clustersize_y",
  title = "Cluster Size Y (OnTrack)",
  xlabel = "size[pixels]"
)

SiPixelPhase1RecHitsErrorX = DefaultHistoTrack.clone(
  enabled=False,
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
  enabled=False,
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

        Specification().groupBy("PXBarrel/PXLayer").saveAll(),
        Specification().groupBy("PXForward/PXDisk").saveAll(),
        StandardSpecification2DProfile,
    
        Specification().groupBy("PXBarrel/PXLayer/Lumisection")
                       .reduce("MEAN")
                       .groupBy("PXBarrel/PXLayer", "EXTEND_X")
                       .save(),

        Specification().groupBy("PXForward/PXDisk/Lumisection")
                       .reduce("MEAN")
                       .groupBy("PXForward/PXDisk", "EXTEND_X")
                       .save(),

        Specification(PerLayer1D).groupBy("PXBarrel/Shell/PXLayer").save(),
        Specification(PerLayer1D).groupBy("PXForward/HalfCylinder/PXRing/PXDisk").save()
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
        src = cms.InputTag("generalTracks"),
        histograms = SiPixelPhase1RecHitsConf,
        geometry = SiPixelPhase1Geometry,
        onlyValidHits = cms.bool(False)

)

SiPixelPhase1RecHitsHarvester = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1RecHitsConf,
        geometry = SiPixelPhase1Geometry
)
