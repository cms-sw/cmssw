import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TEventInfoClient_cff import *
from DQM.L1TMonitorClient.L1TGMTClient_cff import *
from DQM.L1TMonitorClient.L1TGCTClient_cff import *
from DQM.L1TMonitorClient.L1TDTTFClient_cff import *
from DQM.L1TMonitorClient.L1TCSCTFClient_cff import *
from DQM.L1TMonitorClient.L1TDEMONClient_cff import *
from DQM.L1TMonitorClient.L1TRPCTFClient_cff import *

l1tdttfClient.online = cms.untracked.bool(False)

    # use include file for dqmEnv dqmSaver
#from DQMServices.Components.DQMEnvironment_cfi import *
#dqmEnv.subSystemFolder = 'L1T'

#l1tmonitorClient = cms.Sequence(l1tcsctfseqClient*l1tdttpgseqClient*l1trpctfseqClient*l1tdemonseqClient*l1tGctseqClient*l1tEventInfoseqClient*dqmEnv)
#l1tmonitorClient = cms.Sequence(l1tcsctfseqClient*l1tdttpgseqClient*l1trpctfseqClient*l1tdemonseqClient*l1tGctseqClient*l1tEventInfoseqClient)

# Use l1tdttfClient instead of l1tdttpgseqClient
l1tmonitorClient = cms.Sequence(l1tcsctfseqClient*l1tdttfseqClient*l1trpctfseqClient*l1tdemonseqClient*l1tGctseqClient*l1tEventInfoseqClient)

triggerOfflineDQMClient = cms.Sequence(l1tmonitorClient)
