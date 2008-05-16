import FWCore.ParameterSet.Config as cms

#include "DQM/L1TMonitorClient/data/L1THcalClient.cff"
#include "DQM/L1TMonitorClient/data/L1TDTTFClient.cff"
#include "DQM/L1TMonitorClient/data/L1TDTTPGClient.cff"
#include "DQM/L1TMonitorClient/data/L1TdeECALClient.cff"
from DQM.L1TMonitorClient.L1TGMTClient_cff import *
from DQM.L1TMonitorClient.L1TGCTClient_cff import *
#include "DQM/L1TMonitorClient/data/L1TdeECALClient.cff"
from DQM.L1TMonitorClient.L1TCSCTFClient_cff import *
from DQM.L1TMonitorClient.L1TDEMONClient_cff import *
from DQM.L1TMonitorClient.L1TRPCTFClient_cff import *
#    # use include file for dqmEnv dqmSaver
#       include "DQMServices/Components/test/dqm_onlineEnv.cfi"
from DQMServices.Components.DQMEnvironment_cfi import *
#
# END ################################################
#
l1tmonitorClient = cms.Path(l1tgmtseqClient*l1tcsctfseqClient*l1trpctfseqClient*l1tdemonseqClient*l1tGctseqClient*dqmEnv*dqmSaver)

