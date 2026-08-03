import FWCore.ParameterSet.Config as cms

# Second stage clusterizer: merges clusters split by the front end cluster size limit
from RecoLocalTracker.SiPhase2Clusterizer.default_phase2TrackerClusterMerger_cfi import default_phase2TrackerClusterMerger
siPhase2ClustersMerged = default_phase2TrackerClusterMerger.clone(
    src = "siPhase2Clusters"
)
