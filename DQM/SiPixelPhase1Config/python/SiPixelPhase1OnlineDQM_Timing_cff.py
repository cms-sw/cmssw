import FWCore.ParameterSet.Config as cms

from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SuperimoposePlotsInOnlineBlocks=True




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
from DQM.SiPixelPhase1Digis.SiPixelPhase1Digis_cfi import *

# Cluster (track-independent) monitoring
from DQM.SiPixelPhase1Clusters.SiPixelPhase1Clusters_cfi import *

# Track cluster 
from DQM.SiPixelPhase1TrackClusters.SiPixelPhase1TrackClusters_cfi import *
from DQM.SiPixelPhase1TrackResiduals.SiPixelPhase1TrackResiduals_cfi import *


# Raw data errors
from DQM.SiPixelPhase1RawData.SiPixelPhase1RawData_cfi import *

from DQM.SiPixelPhase1Common.SiPixelPhase1GeometryDebug_cfi import *

from DQM.SiPixelPhase1TrackEfficiency.SiPixelPhase1TrackEfficiency_cfi import *

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

SiPixelPhase1TrackClustersAnalyzer_cosmics = SiPixelPhase1TrackClustersAnalyzer.clone()
SiPixelPhase1TrackClustersAnalyzer_cosmics.tracks  = cms.InputTag( "ctfWithMaterialTracksP5" )
SiPixelPhase1TrackClustersAnalyzer_cosmics.VertexCut = cms.untracked.bool(False)

SiPixelPhase1TrackResidualsAnalyzer_cosmics = SiPixelPhase1TrackResidualsAnalyzer.clone()
SiPixelPhase1TrackResidualsAnalyzer_cosmics.Tracks = cms.InputTag( "ctfWithMaterialTracksP5" )
SiPixelPhase1TrackResidualsAnalyzer_cosmics.trajectoryInput = "ctfWithMaterialTracksP5"
SiPixelPhase1TrackResidualsAnalyzer_cosmics.VertexCut = cms.untracked.bool(False)

SiPixelPhase1TrackEfficiencyAnalyzer_cosmics=SiPixelPhase1TrackEfficiencyAnalyzer.clone()
SiPixelPhase1TrackEfficiencyAnalyzer_cosmics.tracks=cms.InputTag( "ctfWithMaterialTracksP5" )

siPixelPhase1OnlineDQM_source_cosmics = cms.Sequence(
   SiPixelPhase1DigisAnalyzer
 + SiPixelPhase1ClustersAnalyzer
 + SiPixelPhase1RawDataAnalyzer
 + SiPixelPhase1TrackClustersAnalyzer_cosmics
 + SiPixelPhase1TrackResidualsAnalyzer_cosmics
)

## Additional settings for pp_run (Phase 0 test)
SiPixelPhase1TrackClustersAnalyzer_pprun = SiPixelPhase1TrackClustersAnalyzer.clone()
SiPixelPhase1TrackClustersAnalyzer_pprun.tracks  = cms.InputTag( "initialStepTracksPreSplitting" )
SiPixelPhase1TrackClustersAnalyzer_pprun.clusterShapeCache = cms.InputTag("siPixelClusterShapeCachePreSplitting")
SiPixelPhase1TrackClustersAnalyzer_pprun.vertices = cms.InputTag('firstStepPrimaryVerticesPreSplitting')
SiPixelPhase1TrackClustersAnalyzer_pprun.VertexCut = cms.untracked.bool(False)

SiPixelPhase1TrackResidualsAnalyzer_pprun = SiPixelPhase1TrackResidualsAnalyzer.clone()
SiPixelPhase1TrackResidualsAnalyzer_pprun.Tracks = cms.InputTag( "initialStepTracksPreSplitting" )
SiPixelPhase1TrackResidualsAnalyzer_pprun.trajectoryInput = "initialStepTracksPreSplitting"
SiPixelPhase1TrackResidualsAnalyzer_pprun.VertexCut = cms.untracked.bool(False)

SiPixelPhase1TrackEfficiencyAnalyzer_pprun=SiPixelPhase1TrackEfficiencyAnalyzer.clone()
SiPixelPhase1TrackEfficiencyAnalyzer_pprun.tracks=cms.InputTag( "initialStepTracksPreSplitting" )
SiPixelPhase1TrackEfficiencyAnalyzer_pprun.VertexCut = cms.untracked.bool(False)

siPixelPhase1OnlineDQM_source_pprun = cms.Sequence(
   SiPixelPhase1DigisAnalyzer
 + SiPixelPhase1ClustersAnalyzer
 + SiPixelPhase1RawDataAnalyzer
 + SiPixelPhase1TrackClustersAnalyzer_pprun
 + SiPixelPhase1TrackResidualsAnalyzer_pprun
 + SiPixelPhase1TrackEfficiencyAnalyzer_pprun
)

