import FWCore.ParameterSet.Config as cms

# description:
# workflow for L1 Trigger Emulator DQM
# used by DQM GUI: DQM/Integration/l1temulator*
# nuno.leonardo@cern.ch 08.02

#global configuration
#from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
#es_prefer_GlobalTag = cms.ESPrefer("PoolDBESSource","GlobalTag")
#note: global tag specification moved top parent _cfg , ie
# : DQM/Integration/pythin/test/l1temulator_dqm_sourceclient-*_cfg.py
# DQM/L1TMonitor/test/test/testEmulMon_cfg.py : test example

#unpacking
from Configuration.StandardSequences.RawToDigi_Data_cff import *

#emulator/comparator
from L1Trigger.HardwareValidation.L1HardwareValidation_cff import *
l1compare.COMPARE_COLLS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]

# temporary fix for L1 GT emulator configuration in hardware validation
valGtDigis.RecordLength = cms.vint32(3, 5)
valGtDigis.AlternativeNrBxBoardDaq = 0x101
valGtDigis.AlternativeNrBxBoardEvm = 0x2

#dqm
from DQM.L1TMonitor.L1TDEMON_cfi import *
from DQM.L1TMonitor.L1TdeECAL_cfi import *
from DQM.L1TMonitor.L1TdeGCT_cfi import *

from DQM.L1TMonitor.L1TdeRCT_cfi import *
l1tderct.rctSourceData = 'gctDigis'
l1tderct.rctSourceEmul = 'valRctDigis'

from DQM.L1TMonitor.L1TdeCSCTF_cfi import *

from DQM.L1TMonitor.l1GtHwValidation_cfi import *

#Note by Nuno: use of edm filters in dqm are discouraged
#-offline they are strictly forbidden
#-online preliminarily allowed if not interferring w other systems

#filter to select "real events"
from HLTrigger.special.HLTTriggerTypeFilter_cfi import *
hltTriggerTypeFilter.SelectedTriggerType = 1

p = cms.Path(
    cms.SequencePlaceholder("RawToDigi")
    *cms.SequencePlaceholder("L1HardwareValidation")
    *(l1demon
      +l1demonecal
      +l1demongct
      +l1decsctf
      +l1GtHwValidation
      #filter goes in the end
      +hltTriggerTypeFilter*l1tderct
      )
    )

