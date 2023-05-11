import FWCore.ParameterSet.Config as cms

import hgcalLayerClustersEE

hgcalLayerClustersHSi = hgcalLayerClustersEE.clone(
  recHits = cms.InputTag('HGCalRecHit', 'HGCHEFRecHits')
)
