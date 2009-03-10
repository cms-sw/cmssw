import FWCore.ParameterSet.Config as cms

from CommonTools.SiStripClusterization.SiStripClusterization_cfi import *

siStripClusterProducer = cms.EDProducer(
    "SiStripClusterProducer",
    SiStripClusterization,
    ProductLabel = cms.InputTag('rawDataCollector'),
    DetSetVectorNew = cms.bool(True),
    )

