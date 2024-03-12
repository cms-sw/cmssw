import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.DefaultClusterizer_cff import *

siStripClusters = cms.EDProducer("SiStripClusterizer",
                               Clusterizer = DefaultClusterizer,
                               DigiProducersList = cms.VInputTag(cms.InputTag('simSiStripDigis','\0'))
                               )
# foo bar baz
# dbVidFWXO8X6d
