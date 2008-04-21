import FWCore.ParameterSet.Config as cms

import copy
from EventFilter.DTRawToDigi.dtunpacker_cfi import *
muonDTDigis = copy.deepcopy(muonDTDigis)
DTRawToDigi = cms.Sequence(muonDTDigis)
muonDTDigis.fedColl = 'rawDataCollector'

