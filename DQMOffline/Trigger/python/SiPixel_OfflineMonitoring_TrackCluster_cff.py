import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMOffline.Trigger.SiPixel_OfflineMonitoring_HistogramManager_cfi import *

# order is important and it should follow ordering in .h !!!
hltSiPixelPhase1TrackClustersOnTrackCharge = hltDefaultHistoTrack.clone(
  name = "charge",
  title = "Corrected Cluster Charge (OnTrack)",
  range_min = 0, range_max = 200e3, range_nbins = 200,
  xlabel = "Charge (electrons)",
  specs = VPSet(
    hltStandardSpecifications1D,
    StandardSpecification2DProfile
  )
)

hltSiPixelPhase1TrackClustersOnTrackSize = hltDefaultHistoTrack.clone(
  name = "size",
  title = "Total Cluster Size (OnTrack)",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "size[pixels]",

  specs = VPSet(
    hltStandardSpecifications1D    
  )
)

hltSiPixelPhase1TrackClustersOnTrackNClusters = hltDefaultHistoTrack.clone(
  name = "clusters_ontrack",
  title = "Clusters_onTrack",
  range_min = 0, range_max = 40000, range_nbins = 4000,
  xlabel = "clusters",
  dimensions = 0,
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer" + "/DetId/Event") 
                   .reduce("COUNT") 
                   .groupBy("PXBarrel/PXLayer")
                   .saveAll(),
    Specification().groupBy("PXForward/PXDisk" + "/DetId/Event") 
                   .reduce("COUNT") 
                   .groupBy("PXForward/PXDisk")
                   .saveAll(),
    StandardSpecificationInclusive_Num,
    StandardSpecificationTrend_Num
  )
)

hltSiPixelPhase1TrackClustersOnTrackPositionB = hltDefaultHistoTrack.clone(
  name = "clusterposition_zphi_ontrack",
  title = "Cluster_onTrack Positions",
  range_min   =  -60, range_max   =  60, range_nbins   = 600,
  range_y_min = -3.2, range_y_max = 3.2, range_y_nbins = 200,
  xlabel = "Global Z", ylabel = "Global \phi",
  dimensions = 2,
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").save(),
#    Specification().groupBy("").save(),
  )
)

hltSiPixelPhase1TrackClustersOnTrackPositionF = hltDefaultHistoTrack.clone(
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

hltSiPixelPhase1TrackClustersOffTrackCharge = hltSiPixelPhase1TrackClustersOnTrackCharge.clone(
    topFolderName = "HLT/Pixel/TrackClusters/OffTrack", 
    enabled = False,
    title = "Cluster Charge"
)
hltSiPixelPhase1TrackClustersOffTrackSize = hltSiPixelPhase1TrackClustersOnTrackSize.clone(
    topFolderName = "HLT/Pixel/TrackClusters/OffTrack",
    enabled = False
)

hltSiPixelPhase1TrackClustersOffTrackNClusters = hltSiPixelPhase1TrackClustersOnTrackNClusters.clone(
    topFolderName = "HLT/Pixel/TrackClusters/OffTrack",
    enabled = False
)

hltSiPixelPhase1TrackClustersOffTrackPositionB = hltSiPixelPhase1TrackClustersOnTrackPositionB.clone(
    topFolderName = "HLT/Pixel/TrackClusters/OffTrack",
    enabled = False
)

hltSiPixelPhase1TrackClustersOffTrackPositionF = hltSiPixelPhase1TrackClustersOnTrackPositionF.clone(
    topFolderName = "HLT/Pixel/TrackClusters/OffTrack",
    enabled = False
)

hltSiPixelPhase1TrackClustersNTracks = hltDefaultHistoTrack.clone(
  name = "ntracks",
  title = "Number of Tracks",
  xlabel = "All - Pixel - BPIX - FPIX",
  range_min = 1, range_max = 5, range_nbins = 4,
  dimensions = 1,
  specs = VPSet(
    Specification().groupBy("").save()
  )
)

hltSiPixelPhase1TrackClustersNTracksInVolume = hltDefaultHistoTrack.clone(
  name = "ntracksinpixvolume",
  title = "Number of Tracks in Pixel fiducial Volume",
  xlabel = "without hits - with hits",
  range_min = 0, range_max = 2, range_nbins = 2,
  dimensions = 1,
  specs = VPSet(
    Specification().groupBy("").save()
  )

)

hltSiPixelPhase1TrackClustersConf = cms.VPSet(
  hltSiPixelPhase1TrackClustersOnTrackCharge,
  hltSiPixelPhase1TrackClustersOnTrackSize,
  hltSiPixelPhase1TrackClustersOnTrackNClusters,
  hltSiPixelPhase1TrackClustersOnTrackPositionB,
  hltSiPixelPhase1TrackClustersOnTrackPositionF,

  hltSiPixelPhase1TrackClustersOffTrackCharge,
  hltSiPixelPhase1TrackClustersOffTrackSize,
  hltSiPixelPhase1TrackClustersOffTrackNClusters,
  hltSiPixelPhase1TrackClustersOffTrackPositionB,
  hltSiPixelPhase1TrackClustersOffTrackPositionF,

  hltSiPixelPhase1TrackClustersNTracks,
  hltSiPixelPhase1TrackClustersNTracksInVolume,
)

hltSiPixelPhase1TrackClustersAnalyzer = cms.EDAnalyzer("SiPixelPhase1TrackClusters",
        clusters   = cms.InputTag("hltSiPixelClusters"),
        tracks     = cms.InputTag("hltTracksMerged"), #hltIter2Merged"
        histograms = hltSiPixelPhase1TrackClustersConf,
        geometry   = hltSiPixelPhase1Geometry
)

hltSiPixelPhase1TrackClustersHarvester = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = hltSiPixelPhase1TrackClustersConf,
        geometry   = hltSiPixelPhase1Geometry
)

