import FWCore.ParameterSet.Config as cms

from DQM.HcalMonitorModule.HcalMonitorModule_cfi import *
hcalOfflineDQMSource = cms.Sequence(hcalMonitor)
hcalMonitor.DataFormatMonitor = True
hcalMonitor.DigiMonitor = True
hcalMonitor.HotCellMonitor = True
hcalMonitor.RecHitMonitor = True
hcalMonitor.TrigPrimMonitor = True
hcalMonitor.PedestalMonitor = False
hcalMonitor.LEDMonitor = False
hcalMonitor.DeadCellMonitor = False
hcalMonitor.MTCCMonitor = False
hcalMonitor.CaloTowerMonitor = False
hcalMonitor.HcalAnalysis = False

