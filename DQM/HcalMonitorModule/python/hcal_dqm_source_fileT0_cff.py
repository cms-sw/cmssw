import FWCore.ParameterSet.Config as cms

from DQM.HcalMonitorModule.HcalMonitorModule_cfi import *
hcalOfflineDQMSource = cms.Sequence(hcalMonitor)
hcalMonitor.DigiOccThresh          = -999999999
hcalMonitor.PedestalsPerChannel    = False
hcalMonitor.PedestalsInFC          = True
hcalMonitor.DataFormatMonitor      = True
hcalMonitor.DigiMonitor            = False
hcalMonitor.RecHitMonitor          = True
hcalMonitor.BeamMonitor            = True
hcalMonitor.TrigPrimMonitor        = True
hcalMonitor.DeadCellMonitor        = False
hcalMonitor.HotCellMonitor         = False
hcalMonitor.PedestalMonitor        = False
hcalMonitor.LEDMonitor             = False
hcalMonitor.MTCCMonitor            = False
hcalMonitor.CaloTowerMonitor       = False
hcalMonitor.HcalAnalysis           = False

