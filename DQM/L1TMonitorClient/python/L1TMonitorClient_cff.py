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


#
# DQM client modules
#

# Bx Timing DQM client module- not available

# LTC DQM client module- not available

# ECAL TPG client DQM module
# not run in L1T - do we need it? FIXME

# HCAL TPG DQM module 
# not run in L1T - do we need it? FIXME

# RCT DQM client module - not available
#from DQM.L1TMonitorClient.L1TRCTClient_cfi import *

# GCT DQM client module 
from DQM.L1TMonitorClient.L1TGCTClient_cfi import *

# DTTPG DQM module 
# not run in L1T - do we need it? FIXME

# DTTF DQM client module 
from DQM.L1TMonitorClient.L1TDTTFClient_cff import *

# CSCTPG DQM module 
# not run in L1T - do we need it? FIXME

# CSCTF DQM client module 
from DQM.L1TMonitorClient.L1TCSCTFClient_cfi import * 

# RPC DQM client module - non-standard name of the module
from DQM.L1TMonitorClient.L1TRPCTFClient_cff import *

# GMT DQM module 
from DQM.L1TMonitorClient.L1TGMTClient_cfi import *

# GT DQM client module - not available 
#from DQM.L1TMonitorClient.L1TGTClient_cfi import *

# L1Extra DQM client module - not available 

# L1 rates DQM client module - not available 

# L1 synchronization DQM client module - not available 

# L1 event info DQM client 
from DQM.L1TMonitorClient.L1TEventInfoClient_cfi import *

#
# other, non pure-L1 stuff
#

# scaler modules (SM and SCAL) - it uses DQM.TrigXMonitor
from DQM.L1TMonitor.L1TMonScalers_cff import *


#
# define sequences 
#

# L1T monitor client sequence 

l1tMonitor = cms.Sequence(
                          bxTiming +
                          l1tLtc +
                          l1tDttf +
                          l1tCsctf + 
                          l1tRpctf +
                          l1tGmt +
                          l1tGt + 
                          l1ExtraDqmSeq +
                          l1tRate +
                          l1tSyc +
                          l1tRctSeq +
                          l1tGctSeq
                          )

# sequence for L1 Trigger DQM client modules on EndPath 
# FIXME clarify why needed on EndPath

l1tMonitorEndPathSeq = cms.Sequence(
                                    l1s +
                                    l1tscalers
                                    )













#l1tmonitorClient = cms.Path(l1tgmtseqClient*l1tcsctfseqClient*l1tdttpgseqClient*l1trpctfseqClient*l1tdemonseqClient*l1tGctseqClient*l1tEventInfoseqClient*dqmEnv*dqmSaver)
l1tmonitorClient = cms.Path(l1tgmtseqClient*l1tgtseqClient*l1tcsctfseqClient*l1tdttfseqClient*l1trpctfseqClient*l1tGctseqClient*l1tRctseqClient*l1tEventInfoseqClient*dqmEnv*dqmSaver)
#l1tmonitorClient = cms.Path(l1tgmtClient*l1tcsctfClient*l1tdttpgClient*l1trpctfClient*l1tdemonseqClient*l1tGctClient*l1tEventInfoseqClient*dqmEnv*dqmSaver)

