import FWCore.ParameterSet.Config as cms

import copy
from EventFilter.HcalRawToDigi.HcalRawToDigi_cfi import *
hcalDigis = copy.deepcopy(hcalDigis)
HcalRawToDigi = cms.Sequence(hcalDigis)
hcalDigis.InputLabel = 'rawDataCollector'

