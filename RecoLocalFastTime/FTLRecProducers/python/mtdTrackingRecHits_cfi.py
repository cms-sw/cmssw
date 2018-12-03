import FWCore.ParameterSet.Config as cms

mtdTrackingRecHits = cms.EDProducer(
    "MTDTrackingRecHitProducer",
    barrelClusters = cms.InputTag('mtdClusters:FTLBarrel'),
    endcapClusters = cms.InputTag('mtdClusters:FTLEndcap'),
)
