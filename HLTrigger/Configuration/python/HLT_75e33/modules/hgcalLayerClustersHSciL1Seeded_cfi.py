import FWCore.ParameterSet.Config as cms

import hgcalLayerClustersEEL1Seeded

hgcalLayerClustersHSciL1Seeded = hgcalLayerClustersEEL1Seeded.clone(
  recHits = cms.InputTag('hltRechitInRegionsHGCAL', 'HGCHEBRecHits'),
  plugin = hgcalLayerClustersEEL1Seeded.plugin.clone(type = cms.string('SciCLUE'))

)
