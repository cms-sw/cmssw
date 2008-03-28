import FWCore.ParameterSet.Config as cms

# raw-to-digi module
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
# zero-suppressor module for raw modes
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
# digi-to-cluster module
#include "RecoLocalTracker/SiStripClusterizer/data/SiStripClusterizer_RealData.cfi"
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
SiStripRawToClusters = cms.Sequence(SiStripDigis*siStripZeroSuppression*siStripClusters)
SiStripDigis.ProductLabel = 'rawDataCollector'
siStripClusters.DigiProducersList = cms.VPSet(cms.PSet(
    DigiLabel = cms.string('ZeroSuppressed'),
    DigiProducer = cms.string('SiStripDigis')
), cms.PSet(
    DigiLabel = cms.string('VirginRaw'),
    DigiProducer = cms.string('siStripZeroSuppression')
), cms.PSet(
    DigiLabel = cms.string('ProcessedRaw'),
    DigiProducer = cms.string('siStripZeroSuppression')
), cms.PSet(
    DigiLabel = cms.string('ScopeMode'),
    DigiProducer = cms.string('siStripZeroSuppression')
))

