import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
#
# David Lange, LLNL
# February 26, 2007
#
# Definition of SiStripRawToDigi sequence
#
#to be true raw to digi sequence -- meanwhile prescaler placeholder
cscRTDPlaceholder = copy.deepcopy(hltPrescaler)
from EventFilter.Configuration.DigiToRaw_cff import *
CSCRawToDigi = cms.Sequence(DigiToRaw*cscRTDPlaceholder)
cscRTDPlaceholder.prescaleFactor = 1

