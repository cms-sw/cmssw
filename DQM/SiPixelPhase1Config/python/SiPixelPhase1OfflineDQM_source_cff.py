import FWCore.ParameterSet.Config as cms

# Pixel Digi Monitoring
from DQM.SiPixelPhase1Digis.SiPixelPhase1Digis_cfi import *
# Cluster (track-independent) monitoring
from DQM.SiPixelPhase1Clusters.SiPixelPhase1Clusters_cfi import *
# RecHit (clusters)
from DQM.SiPixelPhase1RecHits.SiPixelPhase1RecHits_cfi import *
# Residuals
from DQM.SiPixelPhase1TrackResiduals.SiPixelPhase1TrackResiduals_cfi import *
# Clusters ontrack/offtrack (also general tracks)
from DQM.SiPixelPhase1TrackClusters.SiPixelPhase1TrackClusters_cfi import *
# Hit Efficiencies
from DQM.SiPixelPhase1TrackEfficiency.SiPixelPhase1TrackEfficiency_cfi import *

PerModule.enabled = False

siPixelPhase1OfflineDQM_source = cms.Sequence(SiPixelPhase1DigisAnalyzer
                                            + SiPixelPhase1ClustersAnalyzer
                                            + SiPixelPhase1RecHitsAnalyzer
                                            + SiPixelPhase1TrackResidualsAnalyzer
                                            + SiPixelPhase1TrackClustersAnalyzer
                                            + SiPixelPhase1TrackEfficiencyAnalyzer
                                            )
