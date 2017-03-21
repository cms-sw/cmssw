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
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
from DQMOffline.L1Trigger.LegacyL1TriggerDqmOffline_cff import *

#
# DQM L1 Trigger in offline environment
#


from L1Trigger.L1TGlobal.hackConditions_cff import *
from L1Trigger.L1TMuon.hackConditions_cff import *
from L1Trigger.L1TCalorimeter.hackConditions_cff import *
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

from DQM.L1TMonitorClient.L1TStage2CaloLayer2DEClient_cfi import *
from DQM.L1TMonitorClient.L1TStage2MonitorClient_cff import *
# L1T monitor client sequence (system clients and quality tests)
l1TStage2EmulatorClients = cms.Sequence(
                        l1tStage2CaloLayer2DEClient
                        # l1tStage2EmulatorEventInfoClient 
                        )

l1tStage2EmulatorMonitorClient = cms.Sequence(
                        # l1TStage2EmulatorQualityTests +
                        l1TStage2EmulatorClients
                        )

#
# define sequences
#

Stage2l1TriggerOnline = cms.Sequence( 
                               stage2UnpackPath
                                * l1tStage2OnlineDQM
                                * dqmEnvL1T
                               )
                                    
Stage2l1TriggerOffline = cms.Sequence(
                                Stage2l1TriggerOnline
                                 * dqmEnvL1TriggerReco
                                )
 
#
from L1Trigger.Configuration.ValL1Emulator_cff import *

Stage2l1TriggerEmulatorOnline = cms.Sequence(
                                 valHcalTriggerPrimitiveDigis +
                                 Stage2L1HardwareValidation +
                                 l1tStage2EmulatorOnlineDQM +
                                 dqmEnvL1TEMU
                                )

Stage2l1TriggerEmulatorOffline = cms.Sequence(
                                Stage2l1TriggerEmulatorOnline                                
                                )
#

# DQM Offline Step 1 sequence
Stage2l1TriggerDqmOffline = cms.Sequence(
                                Stage2l1TriggerOffline
 #                               * l1tRate_Offline
  #                              * l1tSync_Offline
                                * Stage2l1TriggerEmulatorOffline
                                )                                  

# DQM Offline Step 2 sequence                                 
Stage2l1TriggerDqmOfflineClient = cms.Sequence(
                                l1tStage2EmulatorMonitorClient *
                                l1tStage2MonitorClient
                                )


#replacements for stage2
stage2L1Trigger.toReplaceWith(l1TriggerOnline, Stage2l1TriggerOnline)
stage2L1Trigger.toReplaceWith(l1TriggerOffline, Stage2l1TriggerOffline)
stage2L1Trigger.toReplaceWith(l1TriggerEmulatorOnline, Stage2l1TriggerEmulatorOnline)
stage2L1Trigger.toReplaceWith(l1TriggerEmulatorOffline, Stage2l1TriggerEmulatorOffline)
stage2L1Trigger.toReplaceWith(l1TriggerDqmOffline, Stage2l1TriggerDqmOffline)
stage2L1Trigger.toReplaceWith(l1TriggerDqmOfflineClient, Stage2l1TriggerDqmOfflineClient)




