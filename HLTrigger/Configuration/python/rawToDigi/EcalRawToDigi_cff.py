import FWCore.ParameterSet.Config as cms

import copy
from EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi import *
ecalDigis = copy.deepcopy(ecalEBunpacker)
EcalRawToDigi = cms.Sequence(ecalDigis)
ecalDigis.DoRegional = False
ecalDigis.InputLabel = 'rawDataCollector'

