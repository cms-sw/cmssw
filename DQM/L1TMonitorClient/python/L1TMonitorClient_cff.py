import FWCore.ParameterSet.Config as cms

#include "DQM/L1TMonitorClient/data/L1TDTTPGClient.cff"
from DQM.L1TMonitorClient.L1TEventInfoClient_cff import *
from DQM.L1TMonitorClient.L1TGMTClient_cff import *
from DQM.L1TMonitorClient.L1TGTClient_cff import *
from DQM.L1TMonitorClient.L1TGCTClient_cff import *
from DQM.L1TMonitorClient.L1TRCTClient_cff import *
from DQM.L1TMonitorClient.L1TDTTFClient_cff import *
from DQM.L1TMonitorClient.L1TCSCTFClient_cff import * 
 # out because looking for lots of ME which cannot be found!
from DQM.L1TMonitorClient.L1TDEMONClient_cff import *
from DQM.L1TMonitorClient.L1TRPCTFClient_cff import *
#    # use include file for dqmEnv dqmSaver
#       include "DQMServices/Components/test/dqm_onlineEnv.cfi"
from DQMServices.Components.DQMEnvironment_cfi import *
#
# END ################################################
#

#l1tmonitorClient = cms.Path(l1tgmtseqClient*l1tcsctfseqClient*l1tdttpgseqClient*l1trpctfseqClient*l1tdemonseqClient*l1tGctseqClient*l1tEventInfoseqClient*dqmEnv*dqmSaver)
l1tmonitorClient = cms.Path(l1tgmtseqClient*l1tgtseqClient*l1tcsctfseqClient*l1tdttfseqClient*l1trpctfseqClient*l1tdemonseqClient*l1tGctseqClient*l1tRctseqClient*l1tEventInfoseqClient*dqmEnv*dqmSaver)
#l1tmonitorClient = cms.Path(l1tgmtClient*l1tcsctfClient*l1tdttpgClient*l1trpctfClient*l1tdemonseqClient*l1tGctClient*l1tEventInfoseqClient*dqmEnv*dqmSaver)

