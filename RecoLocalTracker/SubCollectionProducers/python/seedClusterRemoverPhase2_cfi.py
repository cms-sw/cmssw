import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SubCollectionProducers.default_seedClusterRemoverPhase2_cfi import default_seedClusterRemoverPhase2
seedClusterRemoverPhase2 = default_seedClusterRemoverPhase2.clone(
    trajectories = "initialStepSeeds",
    phase2OTClusters = "siPhase2Clusters",
    pixelClusters = "siPixelClusters"
)

