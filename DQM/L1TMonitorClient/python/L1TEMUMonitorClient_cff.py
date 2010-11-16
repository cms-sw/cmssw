# L1 Emulator DQM monitor client 
#
#   authors previous versions - see CVS
#
#   V.M. Ghete 2010-10-22 revised version of L1 emulator DQM

import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1EmulatorQualityTests_cff import *
from DQM.L1TMonitorClient.L1EmulatorErrorFlagClient_cfi import *
from DQM.L1TMonitorClient.L1TEMUEventInfoClient_cff import *


l1EmulatorMonitorClient = cms.Sequence(
                                l1EmulatorQualityTests * 
                                l1EmulatorErrorFlagClient * 
                                l1EmulatorEventInfoClient 
                                )



