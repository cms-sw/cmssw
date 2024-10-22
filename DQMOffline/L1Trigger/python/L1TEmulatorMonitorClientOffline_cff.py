import FWCore.ParameterSet.Config as cms

# adapt the L1TEMUMonitorClient_cff configuration to offline DQM

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


# DQM online L1 Trigger emulator client modules 
from DQM.L1TMonitorClient.L1TEMUMonitorClient_cff import *

# perform offline the quality tests in the clients in endRun only
from DQMOffline.L1Trigger.L1EmulatorQualityTestsOffline_cff import *

# switches for l1EmulatorErrorFlagClient and l1EmulatorEventInfoClient to do and retrieve QTs

l1temuEventInfoClient.runInEventLoop = False
l1temuEventInfoClient.runInEndLumi = False
l1temuEventInfoClient.runInEndRun = True
l1temuEventInfoClient.runInEndJob = False
    
# stage 2
from DQM.L1TMonitorClient.L1TStage2EmulatorMonitorClient_cff import *
l1tStage2EmulatorMonitorClientOffline = cms.Sequence(l1tStage2EmulatorMonitorClient)

