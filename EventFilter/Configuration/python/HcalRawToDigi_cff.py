import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
#
# David Lange, LLNL
# February 26, 2007
#
# Definition of HcalRawToDigi sequence
#
#to be true raw to digi sequence -- meanwhile prescaler placeholder
hcalRTDPlaceholder = copy.deepcopy(hltPrescaler)
from EventFilter.Configuration.DigiToRaw_cff import *
HcalRawToDigi = cms.Sequence(DigiToRaw*hcalRTDPlaceholder)
hcalRTDPlaceholder.prescaleFactor = 1

