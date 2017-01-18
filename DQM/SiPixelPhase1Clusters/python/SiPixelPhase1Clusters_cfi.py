import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1ClustersCharge = DefaultHistoDigiCluster.clone(
  name = "charge",
  title = "Cluster Charge",
  range_min = 0, range_max = 200e3, range_nbins = 200,
  xlabel = "Charge (electrons)",
  
  specs = VPSet(
    #StandardSpecification2DProfile,
    StandardSpecificationPixelmapProfile,
    StandardSpecificationTrend,
    StandardSpecifications1D,
    StandardSpecificationTrend2D
  )
)

SiPixelPhase1ClustersSize = DefaultHistoDigiCluster.clone(
  name = "size",
  title = "Total Cluster Size",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "size[pixels]",
  specs = VPSet(
    #StandardSpecification2DProfile,
    StandardSpecificationPixelmapProfile,
    StandardSpecificationTrend,
    StandardSpecifications1D,
    StandardSpecificationTrend2D
  )
)

SiPixelPhase1ClustersSizeX = DefaultHistoDigiCluster.clone(
  name = "sizeX",
  title = "Cluster Size in X",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "size[pixels]",
  specs = VPSet(
    #StandardSpecification2DProfile,
    #StandardSpecificationPixelmapProfile,
    #StandardSpecificationTrend,
    StandardSpecifications1D,
    #StandardSpecificationTrend2D
  )
)

SiPixelPhase1ClustersSizeY = DefaultHistoDigiCluster.clone(
  name = "sizeY",
  title = "Cluster Size in Y",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "size[pixels]",
  specs = VPSet(
    #StandardSpecification2DProfile,
    #StandardSpecificationPixelmapProfile,
    #StandardSpecificationTrend,
    StandardSpecifications1D,
    #StandardSpecificationTrend2D
  )
)


SiPixelPhase1ClustersNClusters = DefaultHistoDigiCluster.clone(
  name = "clusters",
  title = "Clusters",
  range_min = 0, range_max = 10, range_nbins = 10,
  xlabel = "clusters",
  dimensions = 0,
  specs = VPSet(
    StandardSpecificationOccupancy,
    StandardSpecification2DProfile_Num,
    StandardSpecificationTrend_Num,
    StandardSpecifications1D_Num,
  )
)


SiPixelPhase1ClustersNClustersInclusive = DefaultHistoDigiCluster.clone(
  name = "clusters",
  title = "Clusters",
  range_min = 0, range_max = 2000, range_nbins = 200,
  xlabel = "clusters",
  dimensions = 0,
  specs = VPSet(
    StandardSpecificationInclusive_Num
  )
)


SiPixelPhase1ClustersEventrate = DefaultHistoDigiCluster.clone(
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


SiPixelPhase1ClustersPositionB = DefaultHistoDigiCluster.clone(
  name = "clusterposition_zphi",
  title = "Cluster Positions",
  range_min   =  -60, range_max   =  60, range_nbins   = 600,
  range_y_min = -3.2, range_y_max = 3.2, range_y_nbins = 200,
  xlabel = "Global Z", ylabel = "Global \phi",
  dimensions = 2,
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("").save(),
  )
)

SiPixelPhase1ClustersPositionF = DefaultHistoDigiCluster.clone(
  name = "clusterposition_xy",
  title = "Cluster Positions",
  xlabel = "Global X", ylabel = "Global Y",
  range_min   = -20, range_max   = 20, range_nbins   = 200,
  range_y_min = -20, range_y_max = 20, range_y_nbins = 200,
  dimensions = 2,
  specs = VPSet(
    Specification().groupBy("PXForward/PXDisk").save(),
    Specification().groupBy("PXForward").save(),
    #Specification().groupBy("PXBarrel").save(),
  )
)

SiPixelPhase1ClustersPositionXZ = DefaultHistoDigiCluster.clone(
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

SiPixelPhase1ClustersPositionYZ = DefaultHistoDigiCluster.clone(
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

SiPixelPhase1ClustersSizeVsEta = DefaultHistoDigiCluster.clone(
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

SiPixelPhase1ClustersReadoutCharge = DefaultHistoReadout.clone(
  name = "charge",
  title = "Cluster Charge",
  range_min = 0, range_max = 200e3, range_nbins = 200,
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

SiPixelPhase1ClustersReadoutNClusters = DefaultHistoReadout.clone(
  name = "clusters",
  title = "Clusters",
  range_min = 0, range_max = 10, range_nbins = 10,
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

SiPixelPhase1ClustersConf = cms.VPSet(
  SiPixelPhase1ClustersCharge,
  SiPixelPhase1ClustersSize,
  SiPixelPhase1ClustersSizeX,
  SiPixelPhase1ClustersSizeY,
  SiPixelPhase1ClustersNClusters,
  SiPixelPhase1ClustersNClustersInclusive,
  SiPixelPhase1ClustersEventrate,
  SiPixelPhase1ClustersPositionB,
  SiPixelPhase1ClustersPositionF,
  SiPixelPhase1ClustersPositionXZ,
  SiPixelPhase1ClustersPositionYZ,
  SiPixelPhase1ClustersSizeVsEta,
  SiPixelPhase1ClustersReadoutCharge,
  SiPixelPhase1ClustersReadoutNClusters
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
