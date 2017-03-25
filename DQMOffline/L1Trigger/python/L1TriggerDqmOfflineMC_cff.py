import FWCore.ParameterSet.Config as cms

# L1 Trigger DQM sequence for offline DQM for MC
#
# used by DQM GUI: DQMOffline/Configuration 
#
#
#
# standard RawToDigi sequence and RECO sequence must be run before the L1 Trigger modules, 
# labels from the standard sequence are used as default for the L1 Trigger DQM modules
#
# V.M. Ghete - HEPHY Vienna - 2011-11-17 
#                       

#  L1 Trigger DQM sequence for offline DQM for data                      
from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *

# changes for MC

# do not run the emulator in MC
l1TriggerDqmOffline.remove(l1TriggerEmulatorOnline)                                   


# do not run the emulator client in MC
l1TriggerDqmOfflineClient.remove(l1EmulatorMonitorClient)

import DQMOffline.L1Trigger.L1TriggerDqmOffline_cff as stage2_cffs_
stage2_cffs_.Stage2l1TriggerDqmOffline.remove(stage2_cffs_.l1tStage2EmulatorMonitorClient)
stage2L1Trigger.toReplaceWith(l1TriggerDqmOffline,stage2_cffs_.Stage2l1TriggerDqmOffline)

# correct input tags for MC for the private unpacking
   
dqmGctDigis.inputLabel = 'rawDataCollector'
dqmGtDigis.DaqGtInputTag = 'rawDataCollector'
                                  
