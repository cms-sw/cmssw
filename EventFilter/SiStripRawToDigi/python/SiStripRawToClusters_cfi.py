import FWCore.ParameterSet.Config as cms

from CommonTools.SiStripClusterization.SiStripClusterization_cfi import *
SiStripRawToClustersFacility = cms.EDFilter("SiStripRawToClusters",
    SiStripClusterization,
    ProductLabel = cms.untracked.string('rawDataCollector'),
    ProductInstance = cms.untracked.string('')
)


