import FWCore.ParameterSet.Config as cms

hgcalMultiClusters = cms.EDProducer("HGCalMultiClusterProducer",
    HGCBHInput = cms.InputTag("HGCalRecHit","HGCHEBRecHits"),
    HGCEEInput = cms.InputTag("HGCalRecHit","HGCEERecHits"),
    HGCFHInput = cms.InputTag("HGCalRecHit","HGCHEFRecHits"),
    HGCLayerClusters = cms.InputTag("hgcalLayerClusters"),
    HGCLayerClustersSharing = cms.InputTag("hgcalLayerClusters","sharing"),
    doSharing = cms.bool(False),
    mightGet = cms.optional.untracked.vstring,
    minClusters = cms.uint32(3),
    multiclusterRadii = cms.vdouble(2, 5, 5),
    verbosity = cms.untracked.uint32(3)
)
