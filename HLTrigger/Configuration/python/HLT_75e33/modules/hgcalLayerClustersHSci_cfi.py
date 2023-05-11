import FWCore.ParameterSet.Config as cms

import hgcalLayerClustersEE

hgcalLayerClustersHSci = hgcalLayerClustersEE.clone(
  recHits = cms.InputTag('HGCalRecHit', 'HGCHEBRecHits'),
  plugin = hgcalLayerClustersEE.plugin.clone(type = cms.string('SciCLUE'))
)
  
