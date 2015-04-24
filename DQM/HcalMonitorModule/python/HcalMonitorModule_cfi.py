import FWCore.ParameterSet.Config as cms

feds = [x+700 for x in range(32)] + [1118, 1120, 1122]

hcalMonitor=cms.EDAnalyzer("HcalMonitorModule",
                           debug=cms.untracked.int32(0),
                           online=cms.untracked.bool(False),
                           mergeRuns=cms.untracked.bool(False),
                           enableCleanup=cms.untracked.bool(False),
                           FEDRawDataCollection=cms.untracked.InputTag("rawDataCollector"),
                           UnpackerReport=cms.untracked.InputTag("hcalDigis"),
						   UnpackerReportUTCA=cms.untracked.InputTag("utcaDigis"),
                           subSystemFolder=cms.untracked.string("Hcal/")
#						   FEDs = cms.untracked.vint32(feds)
                           )
