import FWCore.ParameterSet.Config as cms

import copy
from EventFilter.ESRawToDigi.esRawToDigi_cfi import *
ecalPreshowerDigis = copy.deepcopy(esRawToDigi)
EcalESRawToDigi = cms.Sequence(ecalPreshowerDigis)
ecalPreshowerDigis.Label = 'rawDataCollector'

