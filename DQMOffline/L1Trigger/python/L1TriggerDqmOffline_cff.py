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

# DQM online L1 Trigger modules, with offline configuration 
from DQMOffline.L1Trigger.L1TMonitorOffline_cff import *
from DQMOffline.L1Trigger.L1TMonitorClientOffline_cff import *


# DQM offline L1 Trigger versus Reco modules

import DQMServices.Components.DQMEnvironment_cfi
dqmEnvL1TriggerReco = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
dqmEnvL1TriggerReco.subSystemFolder = 'L1T/L1TriggerVsReco'

from DQMOffline.L1Trigger.L1TriggerRecoDQM_cff import *


#
# DQM L1 Trigger Emulator in offline environment
# Run also the L1HwVal producers (L1 Trigger emulators)
#

import DQMServices.Components.DQMEnvironment_cfi
dqmEnvL1TEMU = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
dqmEnvL1TEMU.subSystemFolder = 'L1TEMU'

# DQM Offline Step 1 cfi/cff imports
from DQMOffline.L1Trigger.L1TRate_Offline_cfi import *
from DQMOffline.L1Trigger.L1TSync_Offline_cfi import *
from DQMOffline.L1Trigger.L1TEmulatorMonitorOffline_cff import *  

# DQM Offline Step 2 cfi/cff imports
from DQMOffline.L1Trigger.L1TSync_Harvest_cfi import *
from DQMOffline.L1Trigger.L1TEmulatorMonitorClientOffline_cff import *
from DQMOffline.L1Trigger.L1TEmulatorMonitorClientOffline_cff import *


#
# define sequences 
#

l1TriggerOnline = cms.Sequence(
                               l1tMonitorOnline
                                * dqmEnvL1T
                               )
                                    
l1TriggerOffline = cms.Sequence(
                                l1TriggerOnline
                                 * l1TriggerRecoDQM
                                 * dqmEnvL1TriggerReco
                                )
 
#
 
l1TriggerEmulatorOnline = cms.Sequence(
                                l1HwValEmulatorMonitor
                                * dqmEnvL1TEMU
                                )

l1TriggerEmulatorOffline = cms.Sequence(
                                l1TriggerEmulatorOnline                                
                                )
#

# DQM Offline Step 1 sequence
l1TriggerDqmOffline = cms.Sequence(
                                l1TriggerOffline
                                * l1tRate_Offline
                                * l1tSync_Offline
                                * l1TriggerEmulatorOffline
                                )                                  

# DQM Offline Step 2 sequence                                 
l1TriggerDqmOfflineClient = cms.Sequence(
                                l1tMonitorClient
                                * l1tSync_Harvest
                                * l1EmulatorMonitorClient
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

#

#l1TriggerOffline.remove(l1TriggerOnline)
#l1TriggerOffline.remove(l1TriggerRecoDQM)


# l1tMonitorOnline sequence, defined in DQM/L1TMonitor/python/L1TMonitor_cff.py
#
#l1TriggerOnline.remove(l1tMonitorOnline)
#
l1tMonitorOnline.remove(bxTiming)
#l1tMonitorOnline.remove(l1tDttf)
#l1tMonitorOnline.remove(l1tCsctf) 
#l1tMonitorOnline.remove(l1tRpctf)
#l1tMonitorOnline.remove(l1tGmt)
#l1tMonitorOnline.remove(l1tGt) 
#
#l1ExtraDqmSeq.remove(dqmGctDigis)
#l1ExtraDqmSeq.remove(dqmGtDigis)
#l1ExtraDqmSeq.remove(dqmL1ExtraParticles)
#l1ExtraDqmSeq.remove(l1ExtraDQM)
#l1tMonitorOnline.remove(l1ExtraDqmSeq)
#
l1tMonitorOnline.remove(l1tRate)
l1tMonitorOnline.remove(l1tBPTX)
#l1tMonitorOnline.remove(l1tRctSeq)
#l1tMonitorOnline.remove(l1tGctSeq)

#

#l1TriggerEmulatorOffline.remove(l1TriggerEmulatorOnline)

# l1HwValEmulatorMonitor sequence, defined in DQM/L1TMonitor/python/L1TEmulatorMonitor_cff.py 
#
#l1TriggerEmulatorOnline.remove(l1HwValEmulatorMonitor) 

# L1HardwareValidation producers
#l1HwValEmulatorMonitor.remove(L1HardwareValidation)
#
#l1HwValEmulatorMonitor.remove(l1EmulatorMonitor)


# DQM second step (harvesting)
#

#l1TriggerDqmOfflineClient.remove(l1tMonitorClient)
#l1TriggerDqmOfflineClient.remove(l1EmulatorMonitorClient)

# l1tMonitorClient sequence, defined in DQM/L1TMonitorClient/python/L1TMonitorClient_cff.py
#
#l1tMonitorClient.remove(l1TriggerQualityTests)
#l1tMonitorClient.remove(l1TriggerClients)

# l1TriggerClients sequence, part of l1tMonitorClient sequence

#l1TriggerClients.remove(l1tGctClient)
#l1TriggerClients.remove(l1tDttfClient)
#l1TriggerClients.remove(l1tCsctfClient) 
#l1TriggerClients.remove(l1tRpctfClient)
#l1TriggerClients.remove(l1tGmtClient)
#l1TriggerClients.remove(l1tOccupancyClient)
l1TriggerClients.remove(l1tTestsSummary)
#l1TriggerClients.remove(l1tEventInfoClient)
                              
# l1EmulatorMonitorClient sequence, defined in DQM/L1TMonitorClient/python/L1TEMUMonitorClient_cff.py
#
#l1EmulatorMonitorClient.remove(l1EmulatorQualityTests)
l1EmulatorMonitorClient.remove(l1EmulatorErrorFlagClient)
#l1EmulatorMonitorClient.remove(l1EmulatorEventInfoClient)
