import FWCore.ParameterSet.Config as cms

HGCRecHitCollectionsBlock = cms.PSet(
    HGCRecHitCollections = cms.PSet(
        HGCEEInput = cms.InputTag('HGCalRecHit:HGCEERecHits'),
        HGCFHInput = cms.InputTag('HGCalRecHit:HGCHEFRecHits'),
        HGCBHInput = cms.InputTag('HGCalRecHit:HGCHEBRecHits'),
        )
    )
