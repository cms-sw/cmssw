import FWCore.ParameterSet.Config as cms

from DQM.HcalMonitorClient.HcalMonitorClient_cfi import *
hcalOfflineDQMClient = cms.Sequence(hcalClient)
#hcalClient.plotPedRAW        = True
#hcalClient.DoPerChanTests    = False
hcalClient.baseHtmlDir       = ''
hcalClient.SummaryClient     = True
hcalClient.DataFormatClient  = True
hcalClient.DigiClient        = True
hcalClient.RecHitClient      = True
hcalClient.DeadCellClient    = True
hcalClient.HotCellClient     = True
hcalClient.TrigPrimClient    = False # disabled until we find why it crashes
hcalClient.PedestalClient    = False
hcalClient.LEDClient         = False
hcalClient.CaloTowerClient   = False

