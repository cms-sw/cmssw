import FWCore.ParameterSet.Config as cms

import hgcalLayerClustersEEL1Seeded

hgcalLayerClustersHSiL1Seeded = hgcalLayerClustersEEL1Seeded.clone(
  recHits = cms.InputTag('HGCalRecHit', 'HGCHEFRecHits')
)
