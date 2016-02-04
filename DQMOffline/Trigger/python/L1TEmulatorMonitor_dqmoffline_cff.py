import FWCore.ParameterSet.Config as cms

#global config
#GlobalTag.globaltag = 'CRZT210_V1::All'
#GlobalTag.connect = 'frontier://FrontierProd/CMS_COND_21X_GLOBALTAG'
#es_prefer_GlobalTag = cms.ESPrefer("PoolDBESSource","GlobalTag")

#unpacking
#from Configuration.StandardSequences.RawToDigi_Data_cff import *
#
#emulator/comparator
#from L1Trigger.Configuration.L1Config_cff import *
#from L1Trigger.HardwareValidation.L1HardwareValidation_cff import *
#l1compare.COMPARE_COLLS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
#
from DQM.L1TMonitor.L1TDEMON_cfi import *
from DQM.L1TMonitor.L1TdeECAL_cfi import *
from DQM.L1TMonitor.L1TdeRCT_cfi import *
#
emudqm = cms.Path(
    #cms.SequencePlaceholder("RawToDigi")
    #cms.SequencePlaceholder("L1HardwareValidation")
    cms.SequencePlaceholder("l1demon")
    *cms.SequencePlaceholder("l1demonecal")
    *cms.SequencePlaceholder("l1tderct")
    )
