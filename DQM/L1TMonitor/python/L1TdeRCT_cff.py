import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TGCT_unpack_cff import *
from DQM.L1TMonitor.L1TdeRCT_cfi import *

from HLTrigger.special.HLTTriggerTypeFilter_cfi import *
hltTriggerTypeFilter.SelectedTriggerType = 1

l1tderctpath = cms.Path(hltTriggerTypeFilter*l1GctHwDigis*l1tderct)


