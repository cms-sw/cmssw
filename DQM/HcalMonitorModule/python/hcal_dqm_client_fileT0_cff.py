import FWCore.ParameterSet.Config as cms

from DQM.HcalMonitorClient.HcalMonitorClient_cfi import *
hcalOfflineDQMClient = cms.Sequence(hcalClient)
hcalClient.plotPedRAW = True
hcalClient.DoPerChanTests = False
hcalClient.baseHtmlDir = ''
hcalClient.SummaryClient = True
hcalClient.DataFormatClient = True
hcalClient.DigiClient = False
hcalClient.RecHitClient = False
hcalClient.TrigPrimClient = True
hcalClient.PedestalClient = False
hcalClient.DeadCellClient = False
hcalClient.HotCellClient = False
hcalClient.LEDClient = False
hcalClient.CaloTowerClient = False

