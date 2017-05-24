import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTEGTnPMonitor_cfi import egHLTDQMSelection,egHLTDQMOfflineTnPSource

egammaMonitorHLT = cms.Sequence(
    egHLTDQMSelection*
    egHLTDQMOfflineTnPSource
)
