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

# DQM online L1 Trigger modules 

from DQM.L1TMonitor.L1TMonitor_cff import *
from DQM.L1TMonitorClient.L1TMonitorClient_cff import * 
#
# DTTF to offline configuration
l1tDttf.online = cms.untracked.bool(False) 

# input tag for BXTimining
bxTiming.FedSource = 'rawDataCollector'

#
# for offline, remove the L1TRate which is reading from cms_orcoff_prod, but also requires 
# a hard-coded lxplus path - FIXME check if one can get rid of hard-coded path
# remove also the corresponding client
#
# L1TSync - FIXME - same problems as L1TRate

l1tMonitorOnline.remove(l1tRate)
l1tMonitorClient.remove(l1tTestsSummary)

# FIXME error in l1tDttfClient
l1tMonitorClient.remove(l1tDttfClient)



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

from DQM.L1TMonitor.L1TEmulatorMonitor_cff import *  
from DQM.L1TMonitorClient.L1TEMUMonitorClient_cff import *
    
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

#
                                
l1TriggerDqmOffline = cms.Sequence(
                                l1TriggerOffline 
                                * l1TriggerEmulatorOnline
                                )                                  

# second step in offline environment
                                 
l1TriggerDqmOfflineClient = cms.Sequence(
                                l1tMonitorClient
                                * l1EmulatorMonitorClient
                                )


#
# EMMERGENCY removal of modules
#
# un-comment the module line below to remove the module

# l1tMonitorOnline sequence, defined in DQM/L1TMonitor/python/L1TMonitor_cff.py
#
#l1tMonitorOnline.remove(bxTiming)
l1tMonitorOnline.remove(l1tLtc)
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
#l1tMonitorOnline.remove(l1tRate)
#l1tMonitorOnline.remove(l1tRctSeq)
#l1tMonitorOnline.remove(l1tGctSeq)

# L1HardwareValidation producers
#l1TriggerEmulatorOnline.remove(L1HardwareValidation)                                 
                                    
                                    