import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.dqmHLTFiltersDQMonitor_cfi import *
dqmHLTFiltersDQMonitor.triggerEvent = 'hltTriggerSummaryAOD::HLT'
dqmHLTFiltersDQMonitor.triggerResults = 'TriggerResults::HLT'

from DQMOffline.Trigger.HLTEventInfoClient_cfi import *

hltDqmOffline = cms.Sequence( dqmHLTFiltersDQMonitor * hltEventInfoClient )
