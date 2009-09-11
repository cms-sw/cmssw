import FWCore.ParameterSet.Config as cms

from DQM.HcalMonitorClient.HcalMonitorClient_cfi import *
hcalOfflineDQMClient = cms.Sequence(hcalClient)

hcalClient.baseHtmlDir       = ''
hcalClient.SummaryClient     = True
hcalClient.DataFormatClient  = True
hcalClient.DigiClient        = True
hcalClient.RecHitClient      = True
hcalClient.DeadCellClient    = True
hcalClient.HotCellClient     = True
# Disable trigger primitive client until simulated TP digis included in sequence
hcalClient.TrigPrimClient    = False
hcalClient.PedestalClient    = False
hcalClient.LEDClient         = False
hcalClient.CaloTowerClient   = False

