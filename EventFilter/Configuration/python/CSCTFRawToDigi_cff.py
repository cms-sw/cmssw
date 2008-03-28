import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
#
# David Lange, LLNL
# February 26, 2007
#
# Definition of CSCTFRawToDigi sequence
#
#to be true raw to digi sequence -- meanwhile prescaler placeholder
csctfRTDPlaceholder = copy.deepcopy(hltPrescaler)
from EventFilter.Configuration.DigiToRaw_cff import *
CSCTFRawToDigi = cms.Sequence(DigiToRaw*csctfRTDPlaceholder)
csctfRTDPlaceholder.prescaleFactor = 1

