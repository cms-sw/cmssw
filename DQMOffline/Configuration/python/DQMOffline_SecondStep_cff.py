import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.muonQualityTests_cff import *
from DQMOffline.Ecal.ecal_dqm_client_offline_cff import *
from DQM.HcalMonitorModule.hcal_dqm_client_fileT0_cff import *
DQMOffline_SecondStep = cms.Sequence(ecal_dqm_client_offline*muonQualityTests*hcalOfflineDQMClient)
DQMOffline_SecondStep_woHCAL = cms.Sequence(ecal_dqm_client_offline*muonQualityTests)

