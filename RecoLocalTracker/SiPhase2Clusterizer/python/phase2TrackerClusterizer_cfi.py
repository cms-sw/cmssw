import FWCore.ParameterSet.Config as cms

# Clusterizer options
from RecoLocalTracker.SiPhase2Clusterizer.default_phase2TrackerClusterizer_cfi import default_phase2TrackerClusterizer
siPhase2Clusters = default_phase2TrackerClusterizer.clone(
    src = "mix:Tracker",
    maxClusterSize = 0, # was 8
    maxNumberClusters = 0
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(siPhase2Clusters, src = "mixData:Tracker")

