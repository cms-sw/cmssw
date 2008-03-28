import FWCore.ParameterSet.Config as cms

# relevant FED is 745
from DQM.L1TMonitor.L1TGCT_unpack_cff import *
from DQM.L1TMonitor.L1TRCT_cfi import *
from DQM.L1TMonitor.L1TGCT_cfi import *
l1tgctpath = cms.Path(l1GctHwDigis*l1trct*l1tgct)

