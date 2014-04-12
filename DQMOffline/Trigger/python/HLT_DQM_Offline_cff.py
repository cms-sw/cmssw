import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.GeneralHLTOffline_cfi import *
hltResults.triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT")
hltResults.triggerResultsLabel = cms.InputTag("TriggerResults","","HLT")

#from DQMOffline.Trigger.GeneralHLTOfflineClient_cff import *
from DQMOffline.Trigger.HLTEventInfoClient_cfi import *

#hltDqmOffline = cms.Sequence(hltResults*hltGeneralSeqClient*hltEventInfoClient)
hltDqmOffline = cms.Sequence(hltResults*hltEventInfoClient)

