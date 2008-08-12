import FWCore.ParameterSet.Config as cms

# description:
# workflow for L1 Trigger Emulator DQM
# used by DQM GUI: DQM/Integration/l1temulator*
# nuno.leonardo@cern.ch 08.02

#global configuration
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
es_prefer_GlobalTag = cms.ESPrefer("PoolDBESSource","GlobalTag")
#off-line
#GlobalTag.globaltag = 'CRZT210_V1::All'
#GlobalTag.connect = 'frontier://FrontierProd/CMS_COND_21X_GLOBALTAG'
#on-line
GlobalTag.globaltag = 'CRZT210_V1H::All'
GlobalTag.connect = 'frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG'

#add'n
from Configuration.StandardSequences.Geometry_cff import *

#unpacking
from Configuration.StandardSequences.RawToDigi_Data_cff import *

#emulator/comparator
from L1Trigger.HardwareValidation.L1HardwareValidationGR_cff import *
from L1Trigger.Configuration.L1Config_cff import *
l1compare.COMPARE_COLLS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
#l1compare.DumpMode = -1
#l1compare.VerboseFlag = 1

#dqm
from DQM.L1TMonitor.L1TDEMON_cfi import *
#l1demon.VerboseFlag = -1
#l1demon.HistFile = 'l1demon.root'
from DQM.L1TMonitor.L1TdeECAL_cfi import *
from DQM.L1TMonitor.L1TdeRCT_cfi import *
l1tderct.rctSourceData = 'gctDigis'
l1tderct.rctSourceEmul = 'valRctDigis'

p = cms.Path(
    cms.SequencePlaceholder("RawToDigi")
    *cms.SequencePlaceholder("L1HardwareValidation")
    *cms.SequencePlaceholder("l1demon")
    *cms.SequencePlaceholder("l1demonecal")
    *cms.SequencePlaceholder("l1tderct")
    )

#from L1Trigger.HardwareValidation.hwtest.globrun.deEcal_cff import *
#from L1Trigger.HardwareValidation.hwtest.globrun.deHcal_cff import *
#from L1Trigger.HardwareValidation.hwtest.globrun.deRct_cff import *
#from L1Trigger.HardwareValidation.hwtest.globrun.deGct_cff import *
#from L1Trigger.HardwareValidation.hwtest.globrun.deDt_cff import *
#from L1Trigger.HardwareValidation.hwtest.globrun.deCsc_cff import *
#from L1Trigger.HardwareValidation.hwtest.globrun.deRpc_cff import *
#from L1Trigger.HardwareValidation.hwtest.globrun.deGmt_cff import *
#from L1Trigger.HardwareValidation.hwtest.globrun.deGt_cff import *
