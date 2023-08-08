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
from DQM.SiPixelPhase1Common.SiPixelPhase1Digis_cfi import *

# Cluster (track-independent) monitoring
from DQM.SiPixelPhase1Common.SiPixelPhase1Clusters_cfi import *

# We could overwrite the Harvesters like this, and use the custom() steps to
# perform resetting of histograms.
#SiPixelPhase1ClustersHarvester = cms.EDAnalyzer("SiPixelPhase1OnlineHarvester",
#    histograms = SiPixelPhase1ClustersConf,
#    geometry = SiPixelPhase1Geometry
#)


# Raw data errors
from DQM.SiPixelPhase1Common.SiPixelPhase1RawData_cfi import *
from DQM.SiPixelPhase1Common.SiPixelPhase1DeadFEDChannels_cfi import *

from DQM.SiPixelPhase1Common.SiPixelPhase1GeometryDebug_cfi import *

#Summary maps
from DQM.SiPixelPhase1Summary.SiPixelPhase1Summary_cfi import *

# Track cluster                                                                                                                                                                            
from DQM.SiPixelPhase1Track.SiPixelPhase1TrackClusters_cfi import *
from RecoLocalTracker.SiPixelRecHits.SiPixelTemplateStoreESProducer_cfi import *

SiPixelPhase1TrackClustersOnTrackCorrCharge.enabled=cms.bool(False)
SiPixelPhase1TrackTemplateCorr.enabled=cms.bool(False)
SiPixelPhase1TrackClustersOnTrackCorrChargeOuter.enabled=cms.bool(False)
SiPixelPhase1TrackTemplateCorrOuter.enabled=cms.bool(False)
SiPixelPhase1TrackClustersOnTrackCorrChargeInner.enabled=cms.bool(False)
SiPixelPhase1TrackTemplateCorrInner.enabled=cms.bool(False)
from DQM.SiPixelPhase1Track.SiPixelPhase1TrackResiduals_cfi import *

siPixelPhase1OnlineDQM_source = cms.Sequence(
   SiPixelPhase1DigisAnalyzer
 + SiPixelPhase1DeadFEDChannelsAnalyzer
 + SiPixelPhase1ClustersAnalyzer
 + SiPixelPhase1RawDataAnalyzer
 + SiPixelPhase1TrackClustersAnalyzer
 + SiPixelPhase1TrackResidualsAnalyzer
# + SiPixelPhase1GeometryDebugAnalyzer
 , cms.Task(SiPixelTemplateStoreESProducer)
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

SiPixelPhase1TrackClustersAnalyzer_cosmics = SiPixelPhase1TrackClustersAnalyzer.clone(
    tracks = "ctfWithMaterialTracksP5",
    VertexCut = False
)

SiPixelPhase1TrackResidualsAnalyzer_cosmics = SiPixelPhase1TrackResidualsAnalyzer.clone(
    Tracks = "ctfWithMaterialTracksP5",
    trajectoryInput = "ctfWithMaterialTracksP5",
    VertexCut = False
)

siPixelPhase1OnlineDQM_source_cosmics = cms.Sequence(
   SiPixelPhase1DigisAnalyzer
 + SiPixelPhase1DeadFEDChannelsAnalyzer
 + SiPixelPhase1ClustersAnalyzer
 + SiPixelPhase1RawDataAnalyzer
 + SiPixelPhase1TrackClustersAnalyzer_cosmics
 + SiPixelPhase1TrackResidualsAnalyzer_cosmics,
 cms.Task(SiPixelTemplateStoreESProducer)
)

## Additional settings for pp_run                                                                                                                                         
SiPixelPhase1TrackClustersAnalyzer_pprun = SiPixelPhase1TrackClustersAnalyzer.clone(
    tracks = "initialStepTracksPreSplitting",
    clusterShapeCache = "siPixelClusterShapeCachePreSplitting",
    vertices = 'firstStepPrimaryVerticesPreSplitting',
    VertexCut = False
)

SiPixelPhase1TrackResidualsAnalyzer_pprun = SiPixelPhase1TrackResidualsAnalyzer.clone(
    Tracks = "initialStepTracksPreSplitting",
    trajectoryInput = "initialStepTracksPreSplitting",
    VertexCut = False
)

siPixelPhase1OnlineDQM_source_pprun = cms.Sequence(
   SiPixelPhase1DigisAnalyzer
 + SiPixelPhase1DeadFEDChannelsAnalyzer
 + SiPixelPhase1ClustersAnalyzer
 + SiPixelPhase1RawDataAnalyzer
 + SiPixelPhase1TrackClustersAnalyzer_pprun
 + SiPixelPhase1TrackResidualsAnalyzer_pprun,
 cms.Task(SiPixelTemplateStoreESProducer)
)

siPixelPhase1OnlineDQM_timing_harvesting = siPixelPhase1OnlineDQM_harvesting.copyAndExclude([
 RunQTests_online,
 SiPixelPhase1SummaryOnline,
])
