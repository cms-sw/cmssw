import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMOffline.Trigger.SiPixel_OfflineMonitoring_HistogramManager_cfi import *

# order is important and it should follow ordering in .h !!!
hltSiPixelPhase1ClustersCharge = hltDefaultHistoDigiCluster.clone(
  name = "charge",
  title = "Cluster Charge",
  range_min = 0, range_max = 200e3, range_nbins = 200,
  xlabel = "Charge (electrons)",
  
  specs = VPSet(
    #StandardSpecification2DProfile,
    hltStandardSpecificationPixelmapProfile,
    StandardSpecificationTrend,
#    StandardSpecifications1D,
    hltStandardSpecifications1D,
    StandardSpecificationTrend2D
  )
)

hltSiPixelPhase1ClustersSize = hltDefaultHistoDigiCluster.clone(
  enabled = cms.bool(False), # TO BE CHECKED IF NEEDED
  name = "size",
  title = "Total Cluster Size",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "size[pixels]",
  specs = VPSet(
    #StandardSpecification2DProfile,
    hltStandardSpecificationPixelmapProfile,
    StandardSpecificationTrend,
#    StandardSpecifications1D,
    hltStandardSpecifications1D,
    StandardSpecificationTrend2D
  )
)

hltSiPixelPhase1ClustersSizeX = hltDefaultHistoDigiCluster.clone(
  enabled = cms.bool(False), # TO BE CHECKED IF NEEDED
  name = "sizeX",
  title = "Cluster Size in X",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "size[pixels]",
  specs = VPSet(
    #StandardSpecification2DProfile,
    #StandardSpecificationPixelmapProfile,
    #StandardSpecificationTrend,
#    StandardSpecifications1D,
    hltStandardSpecifications1D,
    #StandardSpecificationTrend2D
  )
)

hltSiPixelPhase1ClustersSizeY = hltDefaultHistoDigiCluster.clone(
  enabled = cms.bool(False), # TO BE CHECKED IF NEEDED
  name = "sizeY",
  title = "Cluster Size in Y",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "size[pixels]",
  specs = VPSet(
    #StandardSpecification2DProfile,
    #StandardSpecificationPixelmapProfile,
    #StandardSpecificationTrend,
#    StandardSpecifications1D,
    hltStandardSpecifications1D,
    #StandardSpecificationTrend2D
  )
)


hltSiPixelPhase1ClustersNClusters = hltDefaultHistoDigiCluster.clone(
  name = "clusters",
  title = "Clusters",
#  range_min = 0, range_max = 40000, range_nbins = 2000,
  range_min = 0, range_max = 40000, range_nbins = 40000,
  xlabel = "clusters",
  dimensions = 0,
  specs = VPSet(
#    StandardSpecificationOccupancy,
#    StandardSpecification2DProfile_Num,
    StandardSpecificationTrend_Num,
    hltStandardSpecifications1D_Num,

  )
)


hltSiPixelPhase1ClustersNClustersInclusive = hltDefaultHistoDigiCluster.clone(
  name = "clusters",
  title = "Clusters",
#  range_min = 0, range_max = 40000, range_nbins = 2000,
  range_min = 0, range_max = 40000, range_nbins = 40000,
  xlabel = "clusters",
  dimensions = 0,
  specs = VPSet(
    StandardSpecificationInclusive_Num,
#    Specification().groupBy("PXAll").save(),
  )
)


hltSiPixelPhase1ClustersEventrate = hltDefaultHistoDigiCluster.clone(
  name = "clustereventrate",
  title = "Number of Events with clusters",
  ylabel = "#Events",
  dimensions = 0,
  specs = VPSet(
    Specification().groupBy("Lumisection")
                   .groupBy("", "EXTEND_X").save(),
    Specification().groupBy("BX")
                   .groupBy("", "EXTEND_X").save()
    )

)


hltSiPixelPhase1ClustersPositionB = hltDefaultHistoDigiCluster.clone(
  name = "clusterposition_zphi",
  title = "Cluster Positions",
  range_min   =  -60, range_max   =  60, range_nbins   = 600,
  range_y_min = -3.2, range_y_max = 3.2, range_y_nbins = 200,
  xlabel = "Global Z", ylabel = "Global \phi",
  dimensions = 2,
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").save(),
#    Specification().groupBy("PXBarrel").save(),
  )
)

hltSiPixelPhase1ClustersPositionF = hltDefaultHistoDigiCluster.clone(
  name = "clusterposition_xy",
  title = "Cluster Positions",
  xlabel = "Global X", ylabel = "Global Y",
  range_min   = -20, range_max   = 20, range_nbins   = 200,
  range_y_min = -20, range_y_max = 20, range_y_nbins = 200,
  dimensions = 2,
  specs = VPSet(
    Specification().groupBy("PXForward/PXDisk").save(),
#    Specification().groupBy("PXForward").save(),
  )
)

hltSiPixelPhase1ClustersPositionXZ = hltDefaultHistoDigiCluster.clone(
  enabled = False, # only for debugging geometry
  name = "clusterposition_xz",
  title = "Cluster Positions",
  xlabel = "Global X", ylabel = "Global Z",
  range_min   = -20, range_max   = 20, range_nbins   = 200,
  range_y_min = -60, range_y_max = 60, range_y_nbins = 1200,
  dimensions = 2,
  specs = VPSet(
  )
)

hltSiPixelPhase1ClustersPositionYZ = hltDefaultHistoDigiCluster.clone(
  enabled = False, # only for debugging geometry
  name = "clusterposition_yz",
  title = "Cluster Positions",
  xlabel = "Global Y", ylabel = "Global Z",
  range_min   = -20, range_max   = 20, range_nbins   = 200,
  range_y_min = -60, range_y_max = 60, range_y_nbins = 1200,
  dimensions = 2,
  specs = VPSet(
  )
)

hltSiPixelPhase1ClustersSizeVsEta = hltDefaultHistoDigiCluster.clone(
  name = "sizeyvseta",
  title = "Cluster Size along Beamline vs. Cluster position #eta",
  xlabel = "Cluster #eta",
  ylabel = "length [pixels]",
  range_min = -3.2, range_max  = 3.2, range_nbins   = 40,
  range_y_min =  0, range_y_max = 40, range_y_nbins = 40,
  dimensions = 2,
  specs = VPSet(
    Specification().groupBy("PXBarrel").save()
  )
)

hltSiPixelPhase1ClustersReadoutCharge = hltDefaultHistoReadout.clone(
  enabled = cms.bool(False),
  name = "charge",
  title = "Cluster Charge",
  range_min = 0, range_max = 200e3, range_nbins = 200, ## e-
  xlabel = "Charge (electrons)",
  specs = VPSet(
    Specification(PerReadout).groupBy("PXBarrel/Shell/Sector").save(),
    Specification(PerReadout).groupBy("PXForward/HalfCylinder").save(),

    Specification(PerReadout).groupBy("PXBarrel/Shell/Sector/OnlineBlock")
                             .groupBy("PXBarrel/Shell/Sector", "EXTEND_Y").save(),
    Specification(PerReadout).groupBy("PXForward/HalfCylinder/OnlineBlock")
                             .groupBy("PXForward/HalfCylinder", "EXTEND_Y").save(),
  )
)

hltSiPixelPhase1ClustersReadoutNClusters = hltDefaultHistoReadout.clone(
  enabled = cms.bool(False),
  name = "clusters",
  title = "Clusters",
  range_min = 0, range_max = 40000, range_nbins = 40000,
  xlabel = "clusters",
  dimensions = 0,
  specs = VPSet(
    Specification(PerReadout).groupBy("PXBarrel/Shell/Sector/DetId/Event").reduce("COUNT")
                             .groupBy("PXBarrel/Shell/Sector").save(),
    Specification(PerReadout).groupBy("PXForward/HalfCylinder/DetId/Event").reduce("COUNT")
                             .groupBy("PXForward/HalfCylinder").save(),

    Specification(PerReadout).groupBy("PXBarrel/Shell/Sector/DetId/Event").reduce("COUNT")
                             .groupBy("PXBarrel/Shell/Sector/Lumisection").reduce("MEAN")
                             .groupBy("PXBarrel/Shell/Sector", "EXTEND_X").save(),
    Specification(PerReadout).groupBy("PXForward/HalfCylinder/DetId/Event").reduce("COUNT")
                             .groupBy("PXForward/HalfCylinder/Lumisection").reduce("MEAN")
                             .groupBy("PXForward/HalfCylinder", "EXTEND_X").save(),
  )
)

hltSiPixelPhase1ClustersPixelToStripRatio = hltDefaultHistoDigiCluster.clone(
  name = "cluster_ratio",
  title = "Pixel to Strip clusters ratio",
  
  xlabel = "ratio",
  dimensions = 1,
  
  specs = VPSet(
    Specification().groupBy("PXAll").save(100, 0, 1), 
    Specification().groupBy("PXAll/Lumisection")
                   .reduce("MEAN") 
                   .groupBy("PXAll", "EXTEND_X")
                   .save(),
    Specification().groupBy("PXAll/BX")
                   .reduce("MEAN") 
                   .groupBy("PXAll", "EXTEND_X")
                   .save(),
  )
)

hltSiPixelPhase1ClustersConf = cms.VPSet(
  hltSiPixelPhase1ClustersCharge,
  hltSiPixelPhase1ClustersSize,
  hltSiPixelPhase1ClustersSizeX,
  hltSiPixelPhase1ClustersSizeY,
  hltSiPixelPhase1ClustersNClusters,
  hltSiPixelPhase1ClustersNClustersInclusive,
  hltSiPixelPhase1ClustersEventrate,
  hltSiPixelPhase1ClustersPositionB,
  hltSiPixelPhase1ClustersPositionF,
  hltSiPixelPhase1ClustersPositionXZ,
  hltSiPixelPhase1ClustersPositionYZ,
  hltSiPixelPhase1ClustersSizeVsEta,
  hltSiPixelPhase1ClustersReadoutCharge,
  hltSiPixelPhase1ClustersReadoutNClusters,
  hltSiPixelPhase1ClustersPixelToStripRatio
)

## Uncomment to add trigger event flag settings
import DQM.SiPixelPhase1Common.TriggerEventFlag_cfi as triggerflag
hltSiPixelPhase1ClustersTriggers = cms.VPSet(
#   triggerflag.genericTriggerEventFlag4HLTdb,
#   triggerflag.genericTriggerEventFlag4L1bd,
)

hltSiPixelPhase1ClustersAnalyzer = cms.EDAnalyzer("SiPixelPhase1Clusters",
        pixelSrc    = cms.InputTag("hltSiPixelClusters"),
        stripSrc    = cms.InputTag("hltSiStripRawToClustersFacility"),
        histograms  = hltSiPixelPhase1ClustersConf,
        geometry    = hltSiPixelPhase1Geometry,
        triggerflag = hltSiPixelPhase1ClustersTriggers,
)

hltSiPixelPhase1ClustersHarvester = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = hltSiPixelPhase1ClustersConf,
        geometry   = hltSiPixelPhase1Geometry
)

