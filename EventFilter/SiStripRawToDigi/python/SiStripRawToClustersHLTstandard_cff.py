import FWCore.ParameterSet.Config as cms

# raw-to-digi module
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
# zero-suppressor module for raw modes
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
# digi-to-cluster module
#include "RecoLocalTracker/SiStripClusterizer/data/SiStripClusterizer_RealData.cfi"
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
SiStripRawToClusters = cms.Sequence(siStripDigis*siStripZeroSuppression*siStripClusters)
siStripDigis.ProductLabel = 'rawDataCollector'
