import FWCore.ParameterSet.Config as cms

trackerGSRecHitTranslator = cms.EDProducer("TrackingRecHitTranslator",
    hitCollectionInputTag = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSRecHits")
)
