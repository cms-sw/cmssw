import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1ClustersCharge = DefaultHisto.clone(
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

SiPixelPhase1ClustersSize = DefaultHisto.clone(
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

SiPixelPhase1ClustersNClusters = DefaultHisto.clone(
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

SiPixelPhase1ClustersEventrate = DefaultHisto.clone(
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

SiPixelPhase1ClustersPositionB = DefaultHisto.clone(
  bookUndefined = False,
  name = "clusterposition",
  title = "Cluster Positions",
  range_min   =  -30, range_max   =  30, range_nbins   = 200,
  range_y_min = -3.2, range_y_max = 3.2, range_y_nbins = 200,
  xlabel = "Global Z", ylabel = "Global \phi",
  dimensions = 2,
  specs = cms.VPSet(
    Specification().groupBy("PXBarrel/PXLayer")
                   .save()
  )
)

SiPixelPhase1ClustersPositionF = DefaultHisto.clone(
  name = "clusterposition",
  title = "Cluster Positions",
  xlabel = "Global X", ylabel = "Global Y",
  range_min   = -20, range_max   = 20, range_nbins   = 200,
  range_y_min = -20, range_y_max = 20, range_y_nbins = 200,
  dimensions = 2,
  specs = cms.VPSet(
    Specification().groupBy("PXForward|PXBarrel/PXDisk|")
                   .save()
  )
)

SiPixelPhase1ClustersSizeVsEta = DefaultHisto.clone(
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

SiPixelPhase1ClustersConf = cms.VPSet(
  SiPixelPhase1ClustersCharge,
  SiPixelPhase1ClustersSize,
  SiPixelPhase1ClustersNClusters,
  SiPixelPhase1ClustersEventrate,
  SiPixelPhase1ClustersPositionB,
  SiPixelPhase1ClustersPositionF,
  SiPixelPhase1ClustersSizeVsEta
)


SiPixelPhase1ClustersAnalyzer = cms.EDAnalyzer("SiPixelPhase1Clusters",
        src = cms.InputTag("siPixelClusters"),
        histograms = SiPixelPhase1ClustersConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1ClustersHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1ClustersConf,
        geometry = SiPixelPhase1Geometry
)
