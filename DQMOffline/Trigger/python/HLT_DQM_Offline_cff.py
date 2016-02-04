import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.FourVectorHLTOffline_cfi import *
hltResults.triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT")
hltResults.triggerResultsLabel = cms.InputTag("TriggerResults","","HLT")

from DQMOffline.Trigger.FourVectorHLTOfflineClient_cff import *
from DQMOffline.Trigger.HLTEventInfoClient_cfi import *

hltDqmOffline = cms.Sequence(hltResults*hltFourVectorSeqClient*hltEventInfoClient)

