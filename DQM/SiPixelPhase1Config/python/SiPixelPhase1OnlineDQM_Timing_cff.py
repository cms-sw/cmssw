import FWCore.ParameterSet.Config as cms

from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SuperimoposePlotsInOnlineBlocks=True
IsOffline.enabled=False


StandardSpecifications1D.append(
        Specification(OverlayCurvesForTiming).groupBy("PXBarrel/PXLayer/OnlineBlock") # per-layer with history for online
                                 .groupBy("PXBarrel/PXLayer", "EXTEND_Y")
                                 .save()
  )
  
StandardSpecifications1D.append(
      Specification(OverlayCurvesForTiming).groupBy("PXForward/PXDisk/OnlineBlock") # per-layer with history for online
                               .groupBy("PXForward/PXDisk", "EXTEND_Y")
                               .save()
  )
  
StandardSpecifications1D.append(
      Specification(OverlayCurvesForTiming).groupBy("PXBarrel/OnlineBlock") # per-layer with history for online
                     .groupBy("PXBarrel", "EXTEND_Y")
                     .save()
  )
StandardSpecifications1D.append(
      Specification(OverlayCurvesForTiming).groupBy("PXForward/OnlineBlock") # per-layer with history for online
                     .groupBy("PXForward", "EXTEND_Y")
                     .save()
  )
  
StandardSpecifications1D_Num.append(
      Specification(OverlayCurvesForTiming).groupBy("DetId/Event") # per-layer with history for online
                               .reduce("COUNT")
                               .groupBy("PXBarrel/PXLayer/OnlineBlock") 
                               .groupBy("PXBarrel/PXLayer", "EXTEND_Y")
                               .save()
  )

StandardSpecifications1D_Num.append(
      Specification(OverlayCurvesForTiming).groupBy("DetId/Event") # per-layer with history for online
                               .reduce("COUNT")
                               .groupBy("PXForward/PXDisk/OnlineBlock") 
                               .groupBy("PXForward/PXDisk", "EXTEND_Y")
                               .save()
  )

#StandardSpecifications1D_Num.append(
#      Specification(OverlayCurvesForTiming).groupBy("PXBarrel/OnlineBlock/PXLayer/Event") # per-layer with history for online
#                     .reduce("COUNT")
#                     .groupBy("PXBarrel/OnlineBlock") 
#                     .groupBy("PXBarrel", "EXTEND_Y")
#                     .save()
#  )


StandardSpecifications1D_Num.append(
Specification(OverlayCurvesForTiming).groupBy("DetId/Event")
                     .reduce("COUNT")
                     .groupBy("PXBarrel/OnlineBlock")
                     .groupBy("PXBarrel", "EXTEND_Y")
                     .save()
)


StandardSpecifications1D_Num.append(
Specification(OverlayCurvesForTiming).groupBy("DetId/Event")
                     .reduce("COUNT")
                     .groupBy("PXForward/OnlineBlock")
                     .groupBy("PXForward", "EXTEND_Y")
                     .save()
)

#StandardSpecifications1D_Num.append(
#      Specification(OverlayCurvesForTiming).groupBy("PXForward/OnlineBlock/PXDisk/Event") # per-layer with history for online
#                     .reduce("COUNT")
#                     .groupBy("PXForward/OnlineBlock") 
#                     .groupBy("PXForward", "EXTEND_Y")
#                     .save()
#  )
#
  
# To Configure Phase1 DQM for Phase0 data
SiPixelPhase1Geometry.upgradePhase = 1
SiPixelPhase1Geometry.onlineblock = 15
# Turn on 'online' harvesting. This has to be set before other configs are 
# loaded (due to how the DefaultHisto PSet is later cloned), therefore it is
# here and not in the harvestng config.
DefaultHisto.perLumiHarvesting = True
DefaultHistoDigiCluster.perLumiHarvesting = True
DefaultHistoSummary.perLumiHarvesting = True
DefaultHistoTrack.perLumiHarvesting = True


# Pixel Digi Monitoring
from DQM.SiPixelPhase1Common.SiPixelPhase1Digis_cfi import *

# Cluster (track-independent) monitoring
from DQM.SiPixelPhase1Common.SiPixelPhase1Clusters_cfi import *
from RecoLocalTracker.SiPixelRecHits.SiPixelTemplateStoreESProducer_cfi import *

# Track cluster 
from DQM.SiPixelPhase1Track.SiPixelPhase1TrackClusters_cfi import *
from DQM.SiPixelPhase1Track.SiPixelPhase1TrackResiduals_cfi import *

# Raw data errors
from DQM.SiPixelPhase1Common.SiPixelPhase1RawData_cfi import *

from DQM.SiPixelPhase1Common.SiPixelPhase1GeometryDebug_cfi import *

from DQM.SiPixelPhase1Track.SiPixelPhase1TrackEfficiency_cfi import *

siPixelPhase1OnlineDQM_source = cms.Sequence(
   SiPixelPhase1DigisAnalyzer
 + SiPixelPhase1ClustersAnalyzer
 + SiPixelPhase1RawDataAnalyzer
 + SiPixelPhase1TrackClustersAnalyzer
 + SiPixelPhase1TrackResidualsAnalyzer
)

siPixelPhase1OnlineDQM_harvesting = cms.Sequence(
   SiPixelPhase1DigisHarvester 
 + SiPixelPhase1ClustersHarvester
 + SiPixelPhase1RawDataHarvester
 + SiPixelPhase1TrackClustersHarvester
 + SiPixelPhase1TrackResidualsHarvester
 + SiPixelPhase1TrackEfficiencyHarvester
)

## Additional settings for cosmic runs 

SiPixelPhase1TrackClustersAnalyzer_cosmics = SiPixelPhase1TrackClustersAnalyzer.clone(
    tracks = "ctfWithMaterialTracksP5", 
    VertexCut = False
)

SiPixelPhase1TrackResidualsAnalyzer_cosmics = SiPixelPhase1TrackResidualsAnalyzer.clone(
    Tracks = "ctfWithMaterialTracksP5",
    trajectoryInput = "ctfWithMaterialTracksP5",
    VertexCut = False
)

SiPixelPhase1TrackEfficiencyAnalyzer_cosmics=SiPixelPhase1TrackEfficiencyAnalyzer.clone(
    tracks = "ctfWithMaterialTracksP5"
)

siPixelPhase1OnlineDQM_source_cosmics = cms.Sequence(
   SiPixelPhase1DigisAnalyzer
 + SiPixelPhase1ClustersAnalyzer
 + SiPixelPhase1RawDataAnalyzer
 + SiPixelPhase1TrackClustersAnalyzer_cosmics
 + SiPixelPhase1TrackResidualsAnalyzer_cosmics
)

## Additional settings for pp_run (Phase 0 test)
SiPixelPhase1TrackClustersAnalyzer_pprun = SiPixelPhase1TrackClustersAnalyzer.clone(
    tracks  = "initialStepTracksPreSplitting",
    clusterShapeCache = "siPixelClusterShapeCachePreSplitting",
    vertices = 'firstStepPrimaryVerticesPreSplitting',
    VertexCut = False
)

SiPixelPhase1TrackResidualsAnalyzer_pprun = SiPixelPhase1TrackResidualsAnalyzer.clone(
    Tracks = "initialStepTracksPreSplitting",
    trajectoryInput = "initialStepTracksPreSplitting",
    VertexCut = False
)

SiPixelPhase1TrackEfficiencyAnalyzer_pprun = SiPixelPhase1TrackEfficiencyAnalyzer.clone(
    tracks = "initialStepTracksPreSplitting",
    VertexCut = False
)

siPixelPhase1OnlineDQM_source_pprun = cms.Sequence(
   SiPixelPhase1DigisAnalyzer
 + SiPixelPhase1ClustersAnalyzer
 + SiPixelPhase1RawDataAnalyzer
 + SiPixelPhase1TrackClustersAnalyzer_pprun
 + SiPixelPhase1TrackResidualsAnalyzer_pprun
 + SiPixelPhase1TrackEfficiencyAnalyzer_pprun
)

