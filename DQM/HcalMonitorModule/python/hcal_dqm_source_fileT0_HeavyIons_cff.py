import FWCore.ParameterSet.Config as cms

# import p+p sequence
from DQM.HcalMonitorModule.hcal_dqm_source_fileT0_cff import *

# modify energy thresholds for HI
hcalHotCellMonitor.energyThreshold = 50.0 # was 10.0
hcalHotCellMonitor.energyThreshold_HF = 200.0 # was 20.0


