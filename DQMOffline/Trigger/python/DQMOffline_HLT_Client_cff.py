import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.FourVectorHLTOfflineClient_cff import *
from DQM.HLTEvF.HLTEventInfoClient_cfi import *

#    # use include file for dqmEnv 
from DQMServices.Components.DQMEnvironment_cfi import *
dqmEnv.subSystemFolder = 'HLTOffline'


hltOfflineDQMClient = cms.Sequence(hltFourVectorSeqClient*hltEventInfoClient*dqmEnv)
