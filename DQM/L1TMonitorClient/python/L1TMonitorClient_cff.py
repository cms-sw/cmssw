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
from DQM.L1TMonitorClient.L1TriggerQualityTests_cff import *

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
from DQM.L1TMonitorClient.L1TStage1Layer2Client_cfi import *

# DTTPG DQM module
# not run in L1T - do we need it? FIXME

# DTTF DQM client module
from DQM.L1TMonitorClient.L1TDTTFClient_cfi import *

# CSCTF DQM client module
from DQM.L1TMonitorClient.L1TCSCTFClient_cfi import *

# RPC DQM client module - non-standard name of the module
from DQM.L1TMonitorClient.L1TRPCTFClient_cfi import *

# GMT DQM module
from DQM.L1TMonitorClient.L1TGMTClient_cfi import *

# GT DQM client module - not available
#from DQM.L1TMonitorClient.L1TGTClient_cfi import *

# L1Extra DQM client module - not available

# L1 rates DQM client module
# L1 synchronization DQM client module
# L1 occupancy DQM client module
from DQM.L1TMonitorClient.L1TOccupancyClient_cff import *
from DQM.L1TMonitorClient.L1TTestsSummary_cff import *

# L1 event info DQM client
from DQM.L1TMonitorClient.L1TEventInfoClient_cfi import *

#
# other, non pure-L1 stuff
#

# scaler modules (SM and SCAL) - it uses DQM.TrigXMonitorClient
from DQM.TrigXMonitorClient.L1TScalersClient_cfi import *
l1tsClient.dqmFolder = cms.untracked.string("L1T/L1Scalers_SM")



#
# define sequences
#

# L1T monitor client sequence (system clients and quality tests)
l1TriggerClients = cms.Sequence(
    l1tGctClient +
    l1tDttfClient +
    l1tCsctfClient +
    l1tRpctfClient +
    l1tGmtClient +
    l1tOccupancyClient +
    l1tTestsSummary +
    l1tEventInfoClient
)

l1TriggerStage1Clients = cms.Sequence(
    l1tStage1Layer2Client +
    l1tDttfClient +
    l1tCsctfClient +
    l1tRpctfClient +
    l1tGmtClient +
    l1tOccupancyClient +
    l1tTestsSummary +
    l1tEventInfoClient
)


l1tMonitorClient = cms.Sequence(
    l1TriggerQualityTests +
    l1TriggerClients
)

l1tMonitorStage1Client = cms.Sequence(
    l1TriggerQualityTests +
    l1TriggerStage1Clients
)


# sequence for L1 Trigger DQM client modules on EndPath
# FIXME clarify why needed on EndPath

l1tMonitorClientEndPathSeq = cms.Sequence(
    l1tsClient
)
