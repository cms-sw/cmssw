import FWCore.ParameterSet.Config as cms

siClusterTranslator = cms.EDProducer("SiClusterTranslator",
    fastTrackerClusterCollectionTag = cms.InputTag("siTrackerGaussianSmearingRecHits", "TrackerClusters")
)
