import FWCore.ParameterSet.Config as cms

# Raw data
from DQM.SiPixelPhase1Common.SiPixelPhase1RawData_cfi import *
# Pixel Digi Monitoring
from DQM.SiPixelPhase1Common.SiPixelPhase1Digis_cfi import *
from DQM.SiPixelPhase1Common.SiPixelPhase1DeadFEDChannels_cfi import *
# Cluster (track-independent) monitoring
from DQM.SiPixelPhase1Common.SiPixelPhase1Clusters_cfi import *
# RecHit (clusters)
from DQM.SiPixelPhase1Track.SiPixelPhase1RecHits_cfi import *
# Residuals
from DQM.SiPixelPhase1Track.SiPixelPhase1TrackResiduals_cfi import *
from DQM.SiPixelPhase1Track.SiPixelPhase1ResidualsExtra_cfi import *
# Clusters ontrack/offtrack (also general tracks)
from DQM.SiPixelPhase1Track.SiPixelPhase1TrackClusters_cfi import *
# Hit Efficiencies
from DQM.SiPixelPhase1Track.SiPixelPhase1TrackEfficiency_cfi import *
# FED/RAW Data
from DQM.SiPixelPhase1Common.SiPixelPhase1RawData_cfi import *
#Summary maps
from DQM.SiPixelPhase1Summary.SiPixelPhase1Summary_cfi import *
#Barycenter plots
from DQM.SiPixelPhase1Summary.SiPixelBarycenter_cfi import *



from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *
from DQM.SiPixelPhase1Track.SiPixelPhase1EfficiencyExtras_cfi import *

PerModule.enabled = False
IsOffline.enabled=True

siPixelPhase1OfflineDQM_source = cms.Sequence(SiPixelPhase1RawDataAnalyzer
                                            + SiPixelPhase1DigisAnalyzer
                                            + SiPixelPhase1DeadFEDChannelsAnalyzer
                                            + SiPixelPhase1ClustersAnalyzer
                                            + SiPixelPhase1RecHitsAnalyzer
                                            + SiPixelPhase1TrackResidualsAnalyzer
                                            + SiPixelPhase1TrackClustersAnalyzer
                                            + SiPixelPhase1TrackEfficiencyAnalyzer
                                            )


#Cosmics config

siPixelPhase1OfflineDQM_source_cosmics = siPixelPhase1OfflineDQM_source.copyAndExclude([
    SiPixelPhase1TrackEfficiencyAnalyzer 
])

SiPixelPhase1TrackResidualsAnalyzer_cosmics = SiPixelPhase1TrackResidualsAnalyzer.clone(
    Tracks = "ctfWithMaterialTracksP5",
    trajectoryInput = "ctfWithMaterialTracksP5",
    VertexCut = False # don't cuts based on the primary vertex position for cosmics
)

siPixelPhase1OfflineDQM_source_cosmics.replace(SiPixelPhase1TrackResidualsAnalyzer,
                                               SiPixelPhase1TrackResidualsAnalyzer_cosmics)

SiPixelPhase1RecHitsAnalyzer_cosmics = SiPixelPhase1RecHitsAnalyzer.clone(
    onlyValidHits = True, # In Cosmics the efficiency plugin will not run, so we monitor only valid hits
    src = "ctfWithMaterialTracksP5",
    VertexCut = False
)

siPixelPhase1OfflineDQM_source_cosmics.replace(SiPixelPhase1RecHitsAnalyzer,
                                               SiPixelPhase1RecHitsAnalyzer_cosmics)

SiPixelPhase1TrackClustersAnalyzer_cosmics = SiPixelPhase1TrackClustersAnalyzer.clone(
    tracks = "ctfWithMaterialTracksP5",
    VertexCut = False
)

siPixelPhase1OfflineDQM_source_cosmics.replace(SiPixelPhase1TrackClustersAnalyzer,
                                               SiPixelPhase1TrackClustersAnalyzer_cosmics)


#heavy ions config

siPixelPhase1OfflineDQM_source_hi = siPixelPhase1OfflineDQM_source.copy()

SiPixelPhase1RecHitsAnalyzer_hi = SiPixelPhase1RecHitsAnalyzer.clone(
    src = "hiGeneralTracks"
)

siPixelPhase1OfflineDQM_source_hi.replace(SiPixelPhase1RecHitsAnalyzer,
                                          SiPixelPhase1RecHitsAnalyzer_hi)

SiPixelPhase1TrackResidualsAnalyzer_hi = SiPixelPhase1TrackResidualsAnalyzer.clone(
    Tracks = "hiGeneralTracks",
    trajectoryInput = "hiRefittedForPixelDQM",
    vertices = "hiSelectedVertex"
)

siPixelPhase1OfflineDQM_source_hi.replace(SiPixelPhase1TrackResidualsAnalyzer,
                                          SiPixelPhase1TrackResidualsAnalyzer_hi)

SiPixelPhase1TrackClustersAnalyzer_hi = SiPixelPhase1TrackClustersAnalyzer.clone(
    tracks = "hiGeneralTracks",
    vertices = "hiSelectedVertex"
)

siPixelPhase1OfflineDQM_source_hi.replace(SiPixelPhase1TrackClustersAnalyzer,
                                               SiPixelPhase1TrackClustersAnalyzer_hi)

SiPixelPhase1TrackEfficiencyAnalyzer_hi = SiPixelPhase1TrackEfficiencyAnalyzer.clone(
    tracks = "hiGeneralTracks",
    primaryvertices = "hiSelectedVertex"
)

siPixelPhase1OfflineDQM_source_hi.replace(SiPixelPhase1TrackEfficiencyAnalyzer,
                                               SiPixelPhase1TrackEfficiencyAnalyzer_hi)
