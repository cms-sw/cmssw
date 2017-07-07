import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTEGTnPMonitor_cfi import egmGsfElectronIDsForDQM,egHLTDQMOfflineTnPSource

egammaMonitorHLT = cms.Sequence(
    egmGsfElectronIDsForDQM*
    egHLTDQMOfflineTnPSource
)
