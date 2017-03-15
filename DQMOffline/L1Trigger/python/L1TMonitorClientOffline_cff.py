import FWCore.ParameterSet.Config as cms

# adapt the L1TMonitorClient_cff configuration to offline DQM

#
# default configuration valid for online DQM
#
# configuration for online DQM
#    process subsystem histograms in endLumi
#    process subsystem histograms in endRun
#
# configuration for offline DQM
#    process subsystem histograms in endRun only
#


# DQM online L1 Trigger client modules
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
if stage2L1Trigger.isChosen():
    from DQM.L1TMonitorClient.L1TStage2MonitorClient_cff import * 
else:
    from DQM.L1TMonitorClient.L1TMonitorClient_cff import *
    # changes for offline environment
    
    # DTTF to offline configuration
    #l1tDttfClient.online = False
    
    # CSCTF client
    #l1tCsctfClient.runInEndLumi = False
    
    # RPC client
    #l1tRpctfClient.runInEndLumi = False
    
    # GMT client
    #l1tGmtClient.runInEndLumi = False
    
    # GCT client
    #l1tGctClient.runInEndLumi = False

