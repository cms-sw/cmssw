import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1TrackClustersOnTrackCharge = DefaultHisto.clone(
  name = "charge",
  title = "Corrected Cluster Charge",
  range_min = 0, range_max = 200e3, range_nbins = 200,
  xlabel = "Charge (electrons)",
  topFolderName = "PixelPhase1/OnTrack",

  specs = cms.VPSet(
    Specification().groupBy("PXBarrel|PXForward/PXLayer|PXDisk").saveAll(),
    StandardSpecification2DProfile
  )
)

SiPixelPhase1TrackClustersOnTrackSize = DefaultHisto.clone(
  name = "size",
  title = "Total Cluster Size",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "size[pixels]",
  topFolderName = "PixelPhase1/OnTrack",

  specs = cms.VPSet(
    Specification().groupBy("PXBarrel|PXForward/PXLayer|PXDisk").saveAll(),
  )
)

SiPixelPhase1TrackClustersOnTrackNClusters = DefaultHisto.clone(
  name = "clusters",
  title = "Clusters",
  range_min = 0, range_max = 10, range_nbins = 10,
  xlabel = "clusters",
  dimensions = 0,
  topFolderName = "PixelPhase1/OnTrack",
  specs = cms.VPSet(
    Specification().groupBy("PXBarrel|PXForward/PXLayer|PXDisk" + "/DetId/Event") 
                   .reduce("COUNT") 
                   .groupBy("PXBarrel|PXForward/PXLayer|PXDisk")
                   .saveAll()
  )
)

SiPixelPhase1TrackClustersOnTrackPositionB = DefaultHisto.clone(
  bookUndefined = False,
  name = "clusterposition_zphi",
  title = "Cluster Positions",
  range_min   =  -60, range_max   =  60, range_nbins   = 600,
  range_y_min = -3.2, range_y_max = 3.2, range_y_nbins = 200,
  xlabel = "Global Z", ylabel = "Global \phi",
  dimensions = 2,
  topFolderName = "PixelPhase1/OnTrack",
  specs = cms.VPSet(
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("").save(),
  )
)

SiPixelPhase1TrackClustersOnTrackPositionF = DefaultHisto.clone(
  bookUndefined = False,
  name = "clusterposition_xy",
  title = "Cluster Positions",
  xlabel = "Global X", ylabel = "Global Y",
  range_min   = -20, range_max   = 20, range_nbins   = 200,
  range_y_min = -20, range_y_max = 20, range_y_nbins = 200,
  dimensions = 2,
  topFolderName = "PixelPhase1/OnTrack",
  specs = cms.VPSet(
    Specification().groupBy("PXForward/PXDisk").save(),
  )
)

SiPixelPhase1TrackClustersOffTrackCharge = \
  SiPixelPhase1TrackClustersOnTrackCharge.clone(topFolderName = "PixelPhase1/OffTrack", 
  title = "Cluster Charge")
SiPixelPhase1TrackClustersOffTrackSize = \
  SiPixelPhase1TrackClustersOnTrackSize.clone(topFolderName = "PixelPhase1/OffTrack")

SiPixelPhase1TrackClustersOffTrackNClusters = \
  SiPixelPhase1TrackClustersOnTrackNClusters.clone(topFolderName = "PixelPhase1/OffTrack")

SiPixelPhase1TrackClustersOffTrackPositionB = \
  SiPixelPhase1TrackClustersOnTrackPositionB.clone(topFolderName = "PixelPhase1/OffTrack")

SiPixelPhase1TrackClustersOffTrackPositionF = \
  SiPixelPhase1TrackClustersOnTrackPositionF.clone(topFolderName = "PixelPhase1/OffTrack")

SiPixelPhase1TrackClustersNTracks = DefaultHisto.clone(
  name = "ntracks",
  title = "Number of Tracks",
  xlabel = "All - Pixel - BPIX - FPIX",
  range_min = 1, range_max = 5, range_nbins = 4,
  dimensions = 1,
  topFolderName = "PixelPhase1/Tracks",
  specs = cms.VPSet(
    Specification().groupBy("").save()
  )
)

SiPixelPhase1TrackClustersNTracksInVolume = DefaultHisto.clone(
  name = "ntracksinpixvolume",
  title = "Number of Tracks in Pixel fiducial Volume",
  xlabel = "without hits - with hits",
  range_min = 0, range_max = 2, range_nbins = 2,
  dimensions = 1,
  topFolderName = "PixelPhase1/Tracks",
  specs = cms.VPSet(
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
        trajectories = cms.InputTag("generalTracks"),
        histograms = SiPixelPhase1TrackClustersConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1TrackClustersHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1TrackClustersConf,
        geometry = SiPixelPhase1Geometry
)
