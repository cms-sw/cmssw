import FWCore.ParameterSet.Config as cms

# RCT data comes from GCT readout
from DQM.L1TMonitor.L1TGCT_unpack_cff import *
from DQM.L1TMonitor.L1TRCT_cfi import *
l1trctpath = cms.Path(l1GctHwDigis*l1trct)

