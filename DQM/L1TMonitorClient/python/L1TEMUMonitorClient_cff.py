import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TDEMONClient_cff import *
from DQM.L1TMonitorClient.L1TEMUEventInfoClient_cff import *
from DQMServices.Components.DQMEnvironment_cfi import *
l1temumonitorClient = cms.Path(l1tdemonseqClient*l1temuEventInfoseqClient*dqmEnv*dqmSaver)
#l1temumonitorClient = cms.Path(l1tdemonseqClient*dqmEnv*dqmSaver)



