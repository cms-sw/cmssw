import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.DefaultClusterizer_cff import *
SiStripRawToClustersFacility = cms.EDProducer("SiStripRawToClusters",
                                              Clusterizer = DefaultClusterizer,
                                              ProductLabel = cms.InputTag('rawDataCollector')
                                              )
