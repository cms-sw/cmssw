import FWCore.ParameterSet.Config as cms



convBremSeeds =cms.EDFilter(
    "ConvBremSeedProducer",
    
    TTRHBuilder = cms.string('WithTrackAngle'),      
    PFClusters = cms.InputTag("particleFlowClusterECAL"),
    pixelRecHits = cms.InputTag("gsPixelRecHits"),                 
    rphirecHits = cms.InputTag("gsStripRecHits","rphiRecHit"),
    matchedrecHits  = cms.InputTag("gsStripRecHits","matchedRecHit"),
    stereorecHits = cms.InputTag("gsStripRecHits","stereoRecHit"),     
    PFRecTrackLabel = cms.InputTag("pfTrackElec")
    )



