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
from DQM.L1TMonitorClient.L1TStage2EMTFQualityTests_cfi import *

# L1 event info DQM client 
from DQM.L1TMonitorClient.L1TEMTFEventInfoClient_cfi import *

#
# define sequences 
#

# L1T monitor client sequence (system clients and quality tests)
l1TEMTFClients = cms.Sequence(
    l1tEMTFEventInfoClient 
    )

l1tEMTFMonitorClient = cms.Sequence(
    l1TStage2EMTFQualityTests +
    l1TEMTFClients
    )
