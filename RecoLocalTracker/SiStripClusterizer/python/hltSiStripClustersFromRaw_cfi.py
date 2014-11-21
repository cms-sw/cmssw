import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.DefaultClusterizer_cff import *
from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *
DefaultClusterizer.MinGoodCharge = cms.double(-1724.)
SiStripClustersFromRawFacility = cms.EDProducer("SiStripClusterizerFromRaw",
                                                onDemand = cms.bool(True),
                                                Clusterizer = DefaultClusterizer,
                                                Algorithms = DefaultAlgorithms,
                                                DoAPVEmulatorCheck = cms.bool(False),
                                                ProductLabel = cms.InputTag('rawDataCollector')
                                                )
