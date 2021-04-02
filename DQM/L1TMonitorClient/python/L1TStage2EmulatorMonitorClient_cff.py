import FWCore.ParameterSet.Config as cms

# L1 Trigger client DQM sequence
#
# used by DQM GUI: DQM/Integration/python/test/l1t_dqm_sourceclient-*_cfg.py
#
# standard RawToDigi sequence must be run before the L1T module, labels
# from the standard sequence are used as default for the L1 DQM modules
# any configuration change in the unpacking must be done in l1t_dqm_sourceclient-*_cfg.py
#
# see CVS for previous authors
#
# V.M. Ghete 2011-05-26 revised version of L1 Trigger client DQM
#

# DQM quality tests
from DQM.L1TMonitorClient.L1TStage2EmulatorQualityTests_cff import *

# Calo trigger layer2 client
from DQM.L1TMonitorClient.L1TStage2CaloLayer2DEClientSummary_cfi import *

# uGMT emulator client
from DQM.L1TMonitorClient.L1TStage2uGMTEmulatorClient_cff import *

# BMTF emulator client
from DQM.L1TMonitorClient.L1TStage2BMTFEmulatorClient_cff import *

# Second BMTF Emulator Client
from DQM.L1TMonitorClient.L1TStage2BMTFSecondEmulatorClient_cff import *

# OMTF emulator client
from DQM.L1TMonitorClient.L1TStage2OMTFEmulatorClient_cff import *

# CSC TPG emulator client
from DQM.L1TMonitorClient.L1TdeCSCTPGClient_cfi import *

# EMTF emulator client
from DQM.L1TMonitorClient.L1TStage2EMTFEmulatorClient_cff import *

# L1 emulator event info DQM client
from DQM.L1TMonitorClient.L1TStage2EmulatorEventInfoClient_cfi import *

## uGT emulator client
from DQM.L1TMonitorClient.L1TStage2uGTEmulatorClient_cff import *

#
# define sequences
#

# L1T monitor client sequence (system clients and quality tests)
l1TStage2EmulatorClients = cms.Sequence(
    l1tStage2CaloLayer2DEClientSummary
    + l1tStage2uGMTEmulatorClient
    + l1tStage2BMTFEmulatorClient
    + l1tStage2BMTFEmulatorSecondClient
    + l1tStage2OMTFEmulatorClient
    + l1tdeCSCTPGClient
    + l1tStage2EMTFEmulatorClient
    + l1tStage2EmulatorEventInfoClient
    + l1tStage2uGTEmulatorClient
)

l1tStage2EmulatorMonitorClient = cms.Sequence(
                        l1TStage2EmulatorQualityTests +
                        l1TStage2EmulatorClients
                        )
