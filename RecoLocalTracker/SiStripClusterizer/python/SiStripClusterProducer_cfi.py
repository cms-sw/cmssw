import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.SiStripClusterization_cfi import *

siStripClusterProducer = cms.EDProducer(
    "SiStripClusterProducer",
    SiStripClusterization,
    ProductLabel = cms.InputTag('rawDataCollector'),
    DetSetVectorNew = cms.bool(True),
    )

