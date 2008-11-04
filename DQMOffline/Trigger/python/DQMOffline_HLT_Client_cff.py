import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.FourVectorHLTOfflineClient_cff import *
#from DQMOffline.Trigger.HLTEventInfoOfflineClient_cfi import *
from DQM.HLTEvF.HLTEventInfoClient_cfi import *

hltOfflineDQMClient = cms.Sequence(hltFourVectorSeqClient*hltEventInfoClient)
