import FWCore.ParameterSet.Config as cms
from copy import deepcopy

zdcMonitor = cms.EDAnalyzer("ZDCMonitorModule",
 debug=cms.untracked.int32(0),
                           online=cms.untracked.bool(False),
                           mergeRuns=cms.untracked.bool(False),
                           enableCleanup=cms.untracked.bool(False),
                           FEDRawDataCollection=cms.untracked.InputTag("rawDataCollector"),
                           UnpackerReport=cms.untracked.InputTag("hcalDigis"),
                           subSystemFolder=cms.untracked.string("Hcal/ZDCMonitor_Hcal/")  # If set to "ZDC", will make a separate folder from Hcal.  (Wouldn't that be nice?)
                            )
