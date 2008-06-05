import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.muonMonitors_cff import *
from DQMOffline.Ecal.ecal_dqm_source_offline_cff import *
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_cff import *
from DQM.HcalMonitorModule.hcal_dqm_source_fileT0_cff import *
from DQMOffline.JetMET.jetMETAnalyzer_cff import *
DQMOffline = cms.Sequence(SiStripDQMTier0*ecal_dqm_source-offline7*muonMonitors*jetMETAnalyzer*hcalOfflineDQMSource)
DQMOffline_woTracker = cms.Sequence(ecal_dqm_source-offline7*muonMonitors*jetMETAnalyzer*hcalOfflineDQMSource)

