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
from DQM.L1TMonitorClient.L1TStage2QualityTests_cff import *


# L1T Objects Ration Timing Plots
from DQM.L1TMonitorClient.L1TObjectsTimingClient_cff import *

# L1 event info DQM client 
from DQM.L1TMonitorClient.L1TStage2EventInfoClient_cfi import *

# BMTF client
from DQM.L1TMonitorClient.L1TStage2BMTFClient_cff import *

# uGMT client
from DQM.L1TMonitorClient.L1TStage2uGMTClient_cff import *

# uGT client
from DQM.L1TMonitorClient.L1TStage2uGTClient_cff import *

# L1 event info DQM client EMTF 
from DQM.L1TMonitorClient.L1TStage2EMTFEventInfoClient_cfi import *

# define sequences 
#

# L1T monitor client sequence (system clients and quality tests)
l1TStage2Clients = cms.Sequence(
                        l1tStage2EventInfoClient
                      + l1tStage2BmtfClient
                      + l1tStage2uGMTClient
                      + l1tStage2uGTClient
                      + l1tStage2EMTFEventInfoClient
                      + l1tObjectsTimingClient
                        )

l1tStage2MonitorClient = cms.Sequence(
                        l1TStage2QualityTests 
                      + l1TStage2Clients 
                        )
