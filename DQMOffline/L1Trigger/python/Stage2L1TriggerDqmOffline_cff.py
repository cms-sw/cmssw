import FWCore.ParameterSet.Config as cms

# L1 Trigger DQM sequence for offline DQM
#
# used by DQM GUI: DQM/Configuration 
#
#
#
# standard RawToDigi sequence and RECO sequence must be run before the L1 Trigger modules, 
# labels from the standard sequence are used as default for the L1 Trigger DQM modules
#
# V.M. Ghete - HEPHY Vienna - 2011-01-02 
#                       
                      

#
# DQM L1 Trigger in offline environment
#

import DQMServices.Components.DQMEnvironment_cfi
dqmEnvL1T = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
dqmEnvL1T.subSystemFolder = 'L1T'

from L1Trigger.L1TGlobal.hackConditions_cff import *
from L1Trigger.L1TMuon.hackConditions_cff import *
from L1Trigger.L1TCalorimeter.hackConditions_cff import *

# DQM online L1 Trigger modules, with offline configuration 
from DQMOffline.L1Trigger.L1TMonitorOffline_cff import *
from DQMOffline.L1Trigger.L1TMonitorClientOffline_cff import *


# DQM offline L1 Trigger versus Reco modules

import DQMServices.Components.DQMEnvironment_cfi
dqmEnvL1TriggerReco = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
dqmEnvL1TriggerReco.subSystemFolder = 'L1T/L1TriggerVsReco'

#
# DQM L1 Trigger Emulator in offline environment
# Run also the L1HwVal producers (L1 Trigger emulators)
#

import DQMServices.Components.DQMEnvironment_cfi
dqmEnvL1TEMU = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
dqmEnvL1TEMU.subSystemFolder = 'L1TEMU'

# DQM Offline Step 1 cfi/cff imports
#from DQMOffline.L1Trigger.L1TRate_Offline_cfi import *
#from DQMOffline.L1Trigger.L1TSync_Offline_cfi import *
#from DQMOffline.L1Trigger.L1TEmulatorMonitorOffline_cff import *  
#l1TdeRCT.rctSourceData = 'gctDigis'

from DQM.L1TMonitor.L1TMonitor_cff import *

# DQM Offline Step 2 cfi/cff imports

from Configuration.StandardSequences.Eras import eras

from DQM.HcalTasks.TPTask import tpTask

#### Test Add
# Filter fat events
#from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel
#hltFatEventFilter = hltHighLevel.clone()
#hltFatEventFilter.throw = cms.bool(False)
#hltFatEventFilter.HLTPaths = cms.vstring('HLT_L1FatEvents_v*')

## This can be used if HLT filter not available in a run
#selfFatEventFilter = cms.EDFilter("HLTL1NumberFilter",
#        invert = cms.bool(False),
#        period = cms.uint32(107),
#        rawInput = cms.InputTag("rawDataCollector"),
#        fedId = cms.int32(1024)
#        )



##Stage 2

from DQM.L1TMonitor.L1TStage2_cff import *

stage2UnpackPath = cms.Sequence(
     l1tCaloLayer1Digis +
     caloStage2Digis +
     bmtfDigis  +
#     BMTFStage2Digis +
     emtfStage2Digis +
     gmtStage2Digis +
     gtStage2Digis 
)

##Stage 2 Emulator

from DQM.L1TMonitor.L1TStage2Emulator_cff import *
#l1tEmulatorMonitorPath = cms.Sequence(
##    hltFatEventFilter +
#    l1tStage2Unpack  +
#    Stage2L1HardwareValidation +
#    l1tStage2EmulatorOnlineDQM
#)

from DQM.L1TMonitorClient.L1TStage2EmulatorMonitorClient_cff import *

#
# define sequences 
#

l1TriggerOnline = cms.Sequence( 
                               stage2UnpackPath
                                * l1tStage2OnlineDQM
                                * dqmEnvL1T
                               )
                                    
l1TriggerOffline = cms.Sequence(
                                l1TriggerOnline
                                 * dqmEnvL1TriggerReco
                                )
 
#
from L1Trigger.Configuration.ValL1Emulator_cff import *

l1TriggerEmulatorOnline = cms.Sequence(
                                 valHcalTriggerPrimitiveDigis +
                                 Stage2L1HardwareValidation +
                                 l1tStage2EmulatorOnlineDQM +
                                 dqmEnvL1TEMU
                                )

l1TriggerEmulatorOffline = cms.Sequence(
                                l1TriggerEmulatorOnline                                
                                )
#

# DQM Offline Step 1 sequence
l1TriggerDqmOffline = cms.Sequence(
                                l1TriggerOffline
 #                               * l1tRate_Offline
  #                              * l1tSync_Offline
                                * l1TriggerEmulatorOffline
                                )                                  

# DQM Offline Step 2 sequence                                 
l1TriggerDqmOfflineClient = cms.Sequence(
                                l1tStage2EmulatorMonitorClient *
                                l1tStage2MonitorClient
                                )


#
#   EMERGENCY   removal of modules or full sequences 
# =============
#
# un-comment the module line below to remove the module or the sequence

#
# NOTE: for offline, remove the L1TRate which is reading from cms_orcoff_prod, but also requires 
# a hard-coded lxplus path - FIXME check if one can get rid of hard-coded path
# remove also the corresponding client
#
# L1TSync - FIXME - same problems as L1TRate


# DQM first step 
#

#l1TriggerDqmOffline.remove(l1TriggerOffline) 
#l1TriggerDqmOffline.remove(l1TriggerEmulatorOffline) 

#Stage 2





