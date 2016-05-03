import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1TrackClustersCharge = DefaultHisto.clone(
  name = "charge",
  title = "Cluster Charge",
  range_min = 0, range_max = 200e3, range_nbins = 200,
  xlabel = "Charge (electrons)",
  
  specs = cms.VPSet(
    Specification().groupBy(DefaultHisto.defaultGrouping)
                   .save()
                   .reduce("MEAN")
                   .groupBy(parent(DefaultHisto.defaultGrouping), "EXTEND_X")
                   .saveAll(),
    Specification().groupBy(parent(DefaultHisto.defaultGrouping)).saveAll(),
    Specification(PerModule).groupBy(DefaultHisto.defaultPerModule).save()
  )
)

SiPixelPhase1TrackClustersSize = DefaultHisto.clone(
  name = "size",
  title = "Total Cluster Size",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "size[pixels]",
  specs = cms.VPSet(
    Specification().groupBy(DefaultHisto.defaultGrouping)
                   .save()
                   .reduce("MEAN")
                   .groupBy(parent(DefaultHisto.defaultGrouping), "EXTEND_X")
                   .saveAll(),
    Specification().groupBy(parent(DefaultHisto.defaultGrouping)).saveAll(),
    Specification(PerModule).groupBy(DefaultHisto.defaultPerModule).save()
  )
)

SiPixelPhase1TrackClustersNClusters = DefaultHisto.clone(
  name = "clusters",
  title = "Clusters",
  range_min = 0, range_max = 10, range_nbins = 10,
  xlabel = "clusters",
  dimensions = 0,
  specs = cms.VPSet(
    Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/DetId/Event") 
                   .reduce("COUNT") 
                   .groupBy(DefaultHisto.defaultGrouping)
                   .save()
                   .reduce("MEAN")
                   .groupBy(parent(DefaultHisto.defaultGrouping), "EXTEND_X")
                   .saveAll(),
    Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/DetId/Event")
                   .reduce("COUNT")
                   .groupBy(parent(DefaultHisto.defaultGrouping))
                   .save(),
    Specification(PerModule).groupBy(DefaultHisto.defaultPerModule.value() + "/Event")
                            .reduce("COUNT")
                            .groupBy(DefaultHisto.defaultPerModule)
                            .save()
  )
)

SiPixelPhase1TrackClustersEventrate = DefaultHisto.clone(
  name = "bigfpixclustereventrate",
  title = "Number of Events with > 180 FPIX clusters",
  xlabel = "Lumisection",
  ylabel = "#Events",
  dimensions = 0,
  specs = cms.VPSet(
    Specification().groupBy("Lumisection")
                   .groupBy("", "EXTEND_X").save()
  )
)

SiPixelPhase1TrackClustersPositionB = DefaultHisto.clone(
  bookUndefined = False,
  name = "clusterposition_zphi",
  title = "Cluster Positions",
  range_min   =  -60, range_max   =  60, range_nbins   = 600,
  range_y_min = -3.2, range_y_max = 3.2, range_y_nbins = 200,
  xlabel = "Global Z", ylabel = "Global \phi",
  dimensions = 2,
  specs = cms.VPSet(
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("").save(),
  )
)

SiPixelPhase1TrackClustersPositionF = DefaultHisto.clone(
  bookUndefined = False,
  name = "clusterposition_xy",
  title = "Cluster Positions",
  xlabel = "Global X", ylabel = "Global Y",
  range_min   = -20, range_max   = 20, range_nbins   = 200,
  range_y_min = -20, range_y_max = 20, range_y_nbins = 200,
  dimensions = 2,
  specs = cms.VPSet(
    Specification().groupBy("PXForward/HalfCylinder/PXDisk").save(),
    Specification().groupBy("PXBarrel").save(),
  )
)

SiPixelPhase1TrackClustersPositionXZ = DefaultHisto.clone(
  enabled = False, # only for debugging geometry
  name = "clusterposition_xz",
  title = "Cluster Positions",
  xlabel = "Global X", ylabel = "Global Z",
  range_min   = -20, range_max   = 20, range_nbins   = 200,
  range_y_min = -60, range_y_max = 60, range_y_nbins = 1200,
  dimensions = 2,
  specs = cms.VPSet(
    Specification().groupBy(DefaultHisto.defaultGrouping)
                   .saveAll(),
  )
)

SiPixelPhase1TrackClustersPositionYZ = DefaultHisto.clone(
  enabled = False, # only for debugging geometry
  name = "clusterposition_yz",
  title = "Cluster Positions",
  xlabel = "Global Y", ylabel = "Global Z",
  range_min   = -20, range_max   = 20, range_nbins   = 200,
  range_y_min = -60, range_y_max = 60, range_y_nbins = 1200,
  dimensions = 2,
  specs = cms.VPSet(
    Specification().groupBy(DefaultHisto.defaultGrouping)
                   .saveAll(),
  )
)

SiPixelPhase1TrackClustersSizeVsEta = DefaultHisto.clone(
  bookUndefined = False, # Barrel only
  name = "sizeyvseta",
  title = "Cluster Size along Beamline vs. Cluster position #eta",
  xlabel = "Cluster #eta",
  ylabel = "length [pixels]",
  range_min = -3.2, range_max  = 3.2, range_nbins   = 40,
  range_y_min =  0, range_y_max = 40, range_y_nbins = 40,
  dimensions = 2,
  specs = cms.VPSet(
    Specification().groupBy("PXBarrel").save()
  )
)

SiPixelPhase1TrackClustersConf = cms.VPSet(
  SiPixelPhase1TrackClustersCharge,
  SiPixelPhase1TrackClustersSize,
  SiPixelPhase1TrackClustersNClusters,
  SiPixelPhase1TrackClustersEventrate,
  SiPixelPhase1TrackClustersPositionB,
  SiPixelPhase1TrackClustersPositionF,
  SiPixelPhase1TrackClustersPositionXZ,
  SiPixelPhase1TrackClustersPositionYZ,
  SiPixelPhase1TrackClustersSizeVsEta
)


SiPixelPhase1TrackClustersAnalyzer = cms.EDAnalyzer("SiPixelPhase1TrackClusters",
        src = cms.InputTag("siPixelClusters"),
        histograms = SiPixelPhase1TrackClustersConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1TrackClustersHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1TrackClustersConf,
        geometry = SiPixelPhase1Geometry
)
