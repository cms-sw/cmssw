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
  
  
# To Configure Phase1 DQM for Phase0 data
SiPixelPhase1Geometry.upgradePhase = 1

#define number of lumis for overlayed plots
SiPixelPhase1Geometry.onlineblock = 150

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

# We could overwrite the Harvesters like this, and use the custom() steps to
# perform resetting of histograms.
#SiPixelPhase1ClustersHarvester = cms.EDAnalyzer("SiPixelPhase1OnlineHarvester",
#    histograms = SiPixelPhase1ClustersConf,
#    geometry = SiPixelPhase1Geometry
#)


# Raw data errors
from DQM.SiPixelPhase1RawData.SiPixelPhase1RawData_cfi import *
from DQM.SiPixelPhase1DeadFEDChannels.SiPixelPhase1DeadFEDChannels_cfi import *

from DQM.SiPixelPhase1Common.SiPixelPhase1GeometryDebug_cfi import *

#Summary maps
from DQM.SiPixelPhase1Summary.SiPixelPhase1Summary_cfi import *

# Track cluster                                                                                                                                                                            
from DQM.SiPixelPhase1TrackClusters.SiPixelPhase1TrackClusters_cfi import *
from DQM.SiPixelPhase1TrackResiduals.SiPixelPhase1TrackResiduals_cfi import *

siPixelPhase1OnlineDQM_source = cms.Sequence(
   SiPixelPhase1DigisAnalyzer
 + SiPixelPhase1DeadFEDChannelsAnalyzer
 + SiPixelPhase1ClustersAnalyzer
 + SiPixelPhase1RawDataAnalyzer
 + SiPixelPhase1TrackClustersAnalyzer
 + SiPixelPhase1TrackResidualsAnalyzer
# + SiPixelPhase1GeometryDebugAnalyzer
)

siPixelPhase1OnlineDQM_harvesting = cms.Sequence(
   SiPixelPhase1DigisHarvester 
 + SiPixelPhase1DeadFEDChannelsHarvester
 + SiPixelPhase1ClustersHarvester
 + SiPixelPhase1RawDataHarvester
 + SiPixelPhase1TrackClustersHarvester
 + SiPixelPhase1TrackResidualsHarvester
 + RunQTests_online
 + SiPixelPhase1SummaryOnline
# + SiPixelPhase1GeometryDebugHarvester
)

## Additional settings for cosmic runs                                                                                                                                                     

SiPixelPhase1TrackClustersAnalyzer_cosmics = SiPixelPhase1TrackClustersAnalyzer.clone()
SiPixelPhase1TrackClustersAnalyzer_cosmics.tracks  = cms.InputTag( "ctfWithMaterialTracksP5" )
SiPixelPhase1TrackClustersAnalyzer_cosmics.VertexCut = cms.untracked.bool(False)

SiPixelPhase1TrackResidualsAnalyzer_cosmics = SiPixelPhase1TrackResidualsAnalyzer.clone()
SiPixelPhase1TrackResidualsAnalyzer_cosmics.Tracks = cms.InputTag( "ctfWithMaterialTracksP5" )
SiPixelPhase1TrackResidualsAnalyzer_cosmics.trajectoryInput = "ctfWithMaterialTracksP5"
SiPixelPhase1TrackResidualsAnalyzer_cosmics.VertexCut = cms.untracked.bool(False)

siPixelPhase1OnlineDQM_source_cosmics = cms.Sequence(
   SiPixelPhase1DigisAnalyzer
 + SiPixelPhase1DeadFEDChannelsAnalyzer
 + SiPixelPhase1ClustersAnalyzer
 + SiPixelPhase1RawDataAnalyzer
 + SiPixelPhase1TrackClustersAnalyzer_cosmics
 + SiPixelPhase1TrackResidualsAnalyzer_cosmics
)

## Additional settings for pp_run                                                                                                                                         
SiPixelPhase1TrackClustersAnalyzer_pprun = SiPixelPhase1TrackClustersAnalyzer.clone()
SiPixelPhase1TrackClustersAnalyzer_pprun.tracks  = cms.InputTag( "initialStepTracksPreSplitting" )
SiPixelPhase1TrackClustersAnalyzer_pprun.clusterShapeCache = cms.InputTag("siPixelClusterShapeCachePreSplitting")
SiPixelPhase1TrackClustersAnalyzer_pprun.vertices = cms.InputTag('firstStepPrimaryVerticesPreSplitting')
SiPixelPhase1TrackClustersAnalyzer_pprun.VertexCut = cms.untracked.bool(False)

SiPixelPhase1TrackResidualsAnalyzer_pprun = SiPixelPhase1TrackResidualsAnalyzer.clone()
SiPixelPhase1TrackResidualsAnalyzer_pprun.Tracks = cms.InputTag( "initialStepTracksPreSplitting" )
SiPixelPhase1TrackResidualsAnalyzer_pprun.trajectoryInput = "initialStepTracksPreSplitting"
SiPixelPhase1TrackResidualsAnalyzer_pprun.VertexCut = cms.untracked.bool(False)

siPixelPhase1OnlineDQM_source_pprun = cms.Sequence(
   SiPixelPhase1DigisAnalyzer
 + SiPixelPhase1DeadFEDChannelsAnalyzer
 + SiPixelPhase1ClustersAnalyzer
 + SiPixelPhase1RawDataAnalyzer
 + SiPixelPhase1TrackClustersAnalyzer_pprun
 + SiPixelPhase1TrackResidualsAnalyzer_pprun
)

siPixelPhase1OnlineDQM_timing_harvesting = siPixelPhase1OnlineDQM_harvesting.copyAndExclude([
 RunQTests_online,
 SiPixelPhase1SummaryOnline,
])
