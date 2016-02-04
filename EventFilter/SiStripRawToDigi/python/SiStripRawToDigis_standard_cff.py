import FWCore.ParameterSet.Config as cms

import copy
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
# raw-to-digi module
#include "EventFilter/SiStripRawToDigi/data/SiStripDigis.cfi"
# zero-suppressor module for raw modes
#include "RecoLocalTracker/SiStripZeroSuppression/data/SiStripZeroSuppression.cfi"
SiStripRawToDigis = cms.Sequence(siStripDigis)

