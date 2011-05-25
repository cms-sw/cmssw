import FWCore.ParameterSet.Config as cms

# SCAL scalers
from DQM.TrigXMonitor.L1TScalersSCAL_cfi import *

# SM scalers
from DQM.TrigXMonitor.L1Scalers_cfi import *
l1s.l1GtData = cms.InputTag("gtDigis","","DQM")
l1s.dqmFolder = cms.untracked.string("L1T/L1Scalers_SM") 
