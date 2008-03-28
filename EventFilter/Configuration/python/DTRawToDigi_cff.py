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
dtRTDPlaceholder = copy.deepcopy(hltPrescaler)
from EventFilter.Configuration.DigiToRaw_cff import *
DTRawToDigi = cms.Sequence(DigiToRaw*dtRTDPlaceholder)
dtRTDPlaceholder.prescaleFactor = 1

