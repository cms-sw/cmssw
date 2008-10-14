import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.muonQualityTests_cff import *
from DQMOffline.Ecal.ecal_dqm_client_offline_cff import *
from DQM.HcalMonitorModule.hcal_dqm_client_fileT0_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_Client_cff import *
from DQMOffline.RecoB.dqmCollector_cff import *
from DQM.SiStripMonitorClient.SiStripClientConfig_Tier0_cff import *
from DQM.DTMonitorClient.dtDQMOfflineClients_cff import *

DQMOffline_SecondStep = cms.Sequence(ecal_dqm_client_offline*muonQualityTests*hcalOfflineDQMClient*sipixelEDAClient*triggerOfflineDQMClient*bTagCollectorSequence*SiStripOfflineDQMClient*dtClients)

