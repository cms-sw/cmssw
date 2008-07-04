import FWCore.ParameterSet.Config as cms

from DQM.HcalMonitorClient.HcalMonitorClient_cfi import *
hcalOfflineDQMClient = cms.Sequence(hcalClient)
hcalClient.plotPedRAW = True
hcalClient.baseHtmlDir = ''
hcalClient.DataFormatClient = True
hcalClient.DigiClient = True
hcalClient.RecHitClient = True
hcalClient.HotCellClient = True
hcalClient.TrigPrimClient = True
hcalClient.LEDClient = False
hcalClient.PedestalClient = False
hcalClient.DeadCellClient = False
hcalClient.CaloTowerClient = False

