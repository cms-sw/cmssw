import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.DefaultClusterizer_cff import *
from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *
SiStripClustersFromRawFacility = cms.EDProducer("SiStripClusterizerFromRaw",
                                                onDemand = cms.bool(False),
                                                Clusterizer = DefaultClusterizer,
                                                Algorithms = DefaultAlgorithms,
                                                DoAPVEmulatorCheck = cms.bool(False),
                                                HybridZeroSuppressed = cms.bool(False),
                                                ProductLabel = cms.InputTag('rawDataCollector')
                                                )
