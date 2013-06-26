import FWCore.ParameterSet.Config as cms

hcalMonitor=cms.EDAnalyzer("HcalMonitorModule",
                           debug=cms.untracked.int32(0),
                           online=cms.untracked.bool(False),
                           mergeRuns=cms.untracked.bool(False),
                           enableCleanup=cms.untracked.bool(False),
                           FEDRawDataCollection=cms.untracked.InputTag("rawDataCollector"),
                           UnpackerReport=cms.untracked.InputTag("hcalDigis"),
                           subSystemFolder=cms.untracked.string("Hcal/"),
                           )
