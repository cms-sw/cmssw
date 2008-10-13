import FWCore.ParameterSet.Config as cms

from DQM.HLTEvF.HLTEventInfoClient_cff import *
from DQM.HLTEvF.HLTMonMuonClient_cff import *

#    # use include file for dqmEnv dqmSaver
#       include "DQMServices/Components/test/dqm_onlineEnv.cfi"
#from DQMServices.Components.DQMEnvironment_cfi import *

hltmonitorClient = cms.Path(hltmonmuonseqClient*hltEventInfoseqClient)




