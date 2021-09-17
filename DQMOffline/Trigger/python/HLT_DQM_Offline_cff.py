import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.hltFiltersDQMonitor_cfi import *
hltFiltersDQMonitor.triggerSummaryAOD = 'hltTriggerSummaryAOD::HLT'
hltFiltersDQMonitor.triggerResults = 'TriggerResults::HLT'

from DQMOffline.Trigger.HLTEventInfoClient_cfi import *

hltDqmOffline = cms.Sequence(hltFiltersDQMonitor*hltEventInfoClient)
