import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TEventInfoClient_cff import *
from DQM.L1TMonitorClient.L1TGMTClient_cff import *
from DQM.L1TMonitorClient.L1TGCTClient_cff import *
from DQM.L1TMonitorClient.L1TDTTFClient_cff import *
from DQM.L1TMonitorClient.L1TCSCTFClient_cff import *
from DQM.L1TMonitorClient.L1TDEMONClient_cff import *
from DQM.L1TMonitorClient.L1TRPCTFClient_cff import *
l1tmonitorClient = cms.Sequence(l1tcsctfseqClient*l1tdttpgseqClient*l1trpctfseqClient*l1tdemonseqClient*l1tGctseqClient*l1tEventInfoseqClient)

triggerOfflineDQMClient = cms.Sequence(l1tmonitorClient)

