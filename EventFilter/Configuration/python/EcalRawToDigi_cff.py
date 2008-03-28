import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
#
# David Lange, LLNL
# February 26, 2007
#
# Definition of EcalRawToDigi sequence
#
#to be true raw to digi sequence -- meanwhile prescaler placeholder
ecalRTDPlaceholder = copy.deepcopy(hltPrescaler)
from EventFilter.Configuration.DigiToRaw_cff import *
EcalRawToDigi = cms.Sequence(DigiToRaw*ecalRTDPlaceholder)
ecalRTDPlaceholder.prescaleFactor = 1

