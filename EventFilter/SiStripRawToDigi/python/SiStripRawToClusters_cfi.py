
# module SiStripRawToClusters -> old Fed9UUtils tkonline sw unpacking libs
# module RawToClusters -> new SistripFEDBuffer CMSSW unpacking libs

import FWCore.ParameterSet.Config as cms

from CommonTools.SiStripClusterization.SiStripClusterization_cfi import *
SiStripRawToClustersFacility = cms.EDProducer("SiStripRawToClusters",
    SiStripClusterization,
    FedRawData = cms.InputTag('rawDataCollector')
)


