import FWCore.ParameterSet.Config as cms

hgcalMultiClusters =  cms.EDProducer(
    "HGCalMultiClusterProducer",
    doSharing = cms.bool(False),
    deltac = cms.vdouble(2.,2.,5.),
    dependSensor = cms.bool(True),
    ecut = cms.double(3.),
    kappa = cms.double(9.),
    multiclusterRadii = cms.vdouble(2.,5.,5.),
    minClusters = cms.uint32(3),
    verbosity = cms.untracked.uint32(3),
    HGCEEInput = cms.InputTag('HGCalRecHit:HGCEERecHits'),
    HGCFHInput = cms.InputTag('HGCalRecHit:HGCHEFRecHits'),
    HGCBHInput = cms.InputTag('HGCalRecHit:HGCHEBRecHits'),
    HGCLayerClusters = cms.InputTag('hgcalLayerClusters:'),
    HGCLayerClustersSharing = cms.InputTag('hgcalLayerClusters:sharing'),
    )
