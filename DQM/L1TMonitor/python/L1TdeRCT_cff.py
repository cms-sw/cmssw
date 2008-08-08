import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TGCT_unpack_cff import *
from DQM.L1TMonitor.L1TdeRCT_cfi import *
l1tderctpath = cms.Path(l1GctHwDigis*l1tderct)


