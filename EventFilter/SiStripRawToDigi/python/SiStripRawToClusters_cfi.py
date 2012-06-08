import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.DefaultClusterizer_cff import *
from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *
SiStripRawToClustersFacility = cms.EDProducer("SiStripRawToClusters",
                                              Clusterizer = DefaultClusterizer,
                                              Algorithms = DefaultAlgorithms,
                                              DoAPVEmulatorCheck = cms.bool(False),
                                              ProductLabel = cms.InputTag('rawDataCollector')
                                              )
