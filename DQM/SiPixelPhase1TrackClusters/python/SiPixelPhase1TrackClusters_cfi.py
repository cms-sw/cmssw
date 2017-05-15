import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1TrackClustersOnTrackCharge = DefaultHistoTrack.clone(
  name = "charge",
  title = "Corrected Cluster Charge (OnTrack)",
  range_min = 0, range_max = 200e3, range_nbins = 200,
  xlabel = "Charge (electrons)",

  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").saveAll(),
    Specification().groupBy("PXForward/PXDisk").saveAll(),
    StandardSpecification2DProfile
  )
)

SiPixelPhase1TrackClustersOnTrackSize = DefaultHistoTrack.clone(
  name = "size",
  title = "Total Cluster Size (OnTrack)",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "size[pixels]",

  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").saveAll(),
    Specification().groupBy("PXForward/PXDisk").saveAll(),
    StandardSpecification2DProfile
  )
)

SiPixelPhase1TrackClustersOnTrackNClusters = DefaultHistoTrack.clone(
  name = "clusters_ontrack",
  title = "Clusters_onTrack",
  range_min = 0, range_max = 10, range_nbins = 10,
  xlabel = "clusters",
  dimensions = 0,

  specs = VPSet(
 #   Specification().groupBy("PXBarrel/PXLayer" + "/DetId/Event") 
 #                  .reduce("COUNT") 
 #                  .groupBy("PXBarrel/PXLayer")
 #                  .saveAll(),
 #   Specification().groupBy("PXForward/PXDisk" + "/DetId/Event") 
 #                  .reduce("COUNT") 
 #                  .groupBy("PXForward/PXDisk")
 #                  .saveAll(),
 #   #StandardSpecificationInclusive_Num,
    StandardSpecificationTrend_Num,
    StandardSpecification2DProfile_Num,

    Specification().groupBy("PXBarrel/PXLayer/Event") #this will produce inclusive counts per Layer/Disk
                             .reduce("COUNT")    
                             .groupBy("PXBarrel/PXLayer")
                             .save(nbins=100, xmin=0, xmax=3000),
    Specification().groupBy("PXForward/PXDisk/Event")
                             .reduce("COUNT")    
                             .groupBy("PXForward/PXDisk/")
                             .save(nbins=100, xmin=0, xmax=3000),

    Specification().groupBy("PXBarrel/Event")
                   .reduce("COUNT")
                   .groupBy("PXBarrel")
                   .save(nbins=200, xmin=0, xmax=10000),
    Specification().groupBy("PXForward/Event")
                   .reduce("COUNT")
                   .groupBy("PXForward")
                   .save(nbins=200, xmin=0, xmax=10000),
    Specification().groupBy("PXAll/Event")
                   .reduce("COUNT")
                   .groupBy("PXAll")
                   .save(nbins=200, xmin=0, xmax=10000),
  )
)

SiPixelPhase1TrackClustersOnTrackPositionB = DefaultHistoTrack.clone(
  name = "clusterposition_zphi_ontrack",
  title = "Cluster_onTrack Positions",
  range_min   =  -60, range_max   =  60, range_nbins   = 600,
  range_y_min = -3.2, range_y_max = 3.2, range_y_nbins = 200,
  xlabel = "Global Z", ylabel = "Global \phi",
  dimensions = 2,
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("").save(),
  )
)

SiPixelPhase1TrackClustersOnTrackPositionF = DefaultHistoTrack.clone(
  name = "clusterposition_xy_ontrack",
  title = "Cluster_onTrack Positions",
  xlabel = "Global X", ylabel = "Global Y",
  range_min   = -20, range_max   = 20, range_nbins   = 200,
  range_y_min = -20, range_y_max = 20, range_y_nbins = 200,
  dimensions = 2,
  specs = VPSet(
    Specification().groupBy("PXForward/PXDisk").save(),
  )
)

SiPixelPhase1TrackClustersOffTrackCharge = \
  SiPixelPhase1TrackClustersOnTrackCharge.clone(topFolderName = "PixelPhase1/OffTrack", 
  enabled = False,
  title = "Cluster Charge")
SiPixelPhase1TrackClustersOffTrackSize = \
  SiPixelPhase1TrackClustersOnTrackSize.clone(topFolderName = "PixelPhase1/OffTrack",
  enabled = False)

SiPixelPhase1TrackClustersOffTrackNClusters = \
  SiPixelPhase1TrackClustersOnTrackNClusters.clone(topFolderName = "PixelPhase1/OffTrack",
  enabled = False)

SiPixelPhase1TrackClustersOffTrackPositionB = \
  SiPixelPhase1TrackClustersOnTrackPositionB.clone(topFolderName = "PixelPhase1/OffTrack",
  enabled = False)

SiPixelPhase1TrackClustersOffTrackPositionF = \
  SiPixelPhase1TrackClustersOnTrackPositionF.clone(topFolderName = "PixelPhase1/OffTrack",
  enabled = False)

SiPixelPhase1TrackClustersNTracks = DefaultHistoTrack.clone(
  name = "ntracks",
  title = "Number of Tracks",
  xlabel = "All - Pixel - BPIX - FPIX",
  range_min = 1, range_max = 5, range_nbins = 4,
  dimensions = 1,
  specs = VPSet(
    Specification().groupBy("").save()
  )
)

SiPixelPhase1TrackClustersNTracksInVolume = DefaultHistoTrack.clone(
  name = "ntracksinpixvolume",
  title = "Number of Tracks in Pixel fiducial Volume",
  xlabel = "without hits - with hits",
  range_min = 0, range_max = 2, range_nbins = 2,
  dimensions = 1,
  specs = VPSet(
    Specification().groupBy("").save()
  )

)

SiPixelPhase1TrackClustersConf = cms.VPSet(
  SiPixelPhase1TrackClustersOnTrackCharge,
  SiPixelPhase1TrackClustersOnTrackSize,
  SiPixelPhase1TrackClustersOnTrackNClusters,
  SiPixelPhase1TrackClustersOnTrackPositionB,
  SiPixelPhase1TrackClustersOnTrackPositionF,

  SiPixelPhase1TrackClustersOffTrackCharge,
  SiPixelPhase1TrackClustersOffTrackSize,
  SiPixelPhase1TrackClustersOffTrackNClusters,
  SiPixelPhase1TrackClustersOffTrackPositionB,
  SiPixelPhase1TrackClustersOffTrackPositionF,

  SiPixelPhase1TrackClustersNTracks,
  SiPixelPhase1TrackClustersNTracksInVolume,
)


SiPixelPhase1TrackClustersAnalyzer = cms.EDAnalyzer("SiPixelPhase1TrackClusters",
        clusters = cms.InputTag("siPixelClusters"),
        tracks = cms.InputTag("generalTracks"),
        histograms = SiPixelPhase1TrackClustersConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1TrackClustersHarvester = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1TrackClustersConf,
        geometry = SiPixelPhase1Geometry
)
