import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.muonMonitors_cff import *
from DQMOffline.Ecal.ecal_dqm_source_offline_cff import *
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_cff import *
DQMOffline = cms.Sequence(SiStripDQMTier0*ecal_dqm_source-offline*muonMonitors)
DQMOffline_woCSC = cms.Sequence(SiStripDQMTier0*ecal_dqm_source-offline*muonMonitors_woCSC)
DQMOffline_woTrackerAndCSC = cms.Sequence(ecal_dqm_source-offline*muonMonitors_woCSC)
DQMOffline_woTracker = cms.Sequence(ecal_dqm_source-offline*muonMonitors)

