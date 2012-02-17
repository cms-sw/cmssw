import FWCore.ParameterSet.Config as cms

# import p+p sequence
from DQM.HcalMonitorModule.hcal_dqm_source_fileT0_cff import *

# modify energy thresholds for HI
hcalHotCellMonitor.energyThreshold = 50.0 # was 10.0
hcalHotCellMonitor.energyThreshold_HF = 200.0 # was 20.0



# HEAVY ION UPDATE ON 18 NOV 2011 TO CHANGE HF DIGI SIZE
# HF digis have 10 TS, just as HBHE and HO in HEAVY ION RUNNING

# These settings used in tag 'HeavyIonTag_for_442p6_v2loose', but are disabled in current HEAD
hcalDigiMonitor.minDigiSizeHF = 0
hcalDigiMonitor.maxDigiSizeHF = 10
