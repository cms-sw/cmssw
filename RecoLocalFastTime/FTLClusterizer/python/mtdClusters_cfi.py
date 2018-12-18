import FWCore.ParameterSet.Config as cms

mtdClusters = cms.EDProducer("MTDClusterProducer",
    srcBarrel = cms.InputTag("mtdRecHits:FTLBarrel"),
    srcEndcap = cms.InputTag("mtdRecHits:FTLEndcap"),
    BarrelClusterName = cms.string('FTLBarrel'),
    EndcapClusterName = cms.string('FTLEndcap'),
    ClusterMode = cms.string('MTDThresholdClusterizer'),
    HitThreshold = cms.double(0.),
    SeedThreshold = cms.double(0.),
    ClusterThreshold = cms.double(0.)
)
