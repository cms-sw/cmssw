import FWCore.ParameterSet.Config as cms

from CondTools.DQM.DQMReferenceHistogramRootFileEventSetupAnalyzer_cfi import *
from DQMServices.Components.DQMMessageLoggerClient_cfi import *
from DQMServices.Components.DQMDcsInfoClient_cfi import *

from DQMOffline.Ecal.ecal_dqm_client_offline_cff import *
from DQM.HcalMonitorModule.hcal_dqm_client_fileT0_cff import *
from DQM.SiStripMonitorClient.SiStripClientConfig_Tier0_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff import *
from DQM.DTMonitorClient.dtDQMOfflineClients_cff import *
from DQM.RPCMonitorClient.RPCTier0Client_cff import *
from DQM.CSCMonitorModule.csc_dqm_offlineclient_collisions_cff import *
from DQM.EcalPreshowerMonitorClient.es_dqm_client_offline_cff import *
from DQMServices.Components.DQMFEDIntegrityClient_cff import *

DQMOffline_SecondStep_PreDPG = cms.Sequence( dqmDcsInfoClient *
#                                             ecal_dqm_client_offline *
#                                             hcalOfflineDQMClient *
                                             SiStripOfflineDQMClient *
                                             PixelOfflineDQMClientWithDataCertification *
                                             dtClients *
                                             rpcTier0Client *
                                             cscOfflineCollisionsClients 
# *
#                                             es_dqm_client_offline *
#                                             dqmFEDIntegrityClient 
)

DQMOffline_SecondStepDPG = cms.Sequence( dqmRefHistoRootFileGetter *
                                         DQMOffline_SecondStep_PreDPG *
                                         DQMMessageLoggerClient )

from DQMOffline.Muon.muonQualityTests_cff import *
from DQMOffline.EGamma.photonOfflineDQMClient_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_Client_cff import *
from DQMOffline.Trigger.DQMOffline_HLT_Client_cff import *
from DQMOffline.RecoB.dqmCollector_cff import *

#DQMOffline_SecondStep_PrePOG = cms.Sequence( muonQualityTests *
#                                             photonOfflineDQMClient *
#                                             triggerOfflineDQMClient *
#                                             hltOfflineDQMClient *
#                                             bTagCollectorSequence )
DQMOffline_SecondStep_PrePOG = cms.Sequence( muonQualityTests )

DQMOffline_SecondStepPOG = cms.Sequence( dqmRefHistoRootFileGetter *
                                         DQMOffline_SecondStep_PrePOG *
                                         DQMMessageLoggerClient )

DQMOffline_SecondStep = cms.Sequence( dqmRefHistoRootFileGetter *
                                      DQMOffline_SecondStep_PreDPG *
                                      DQMOffline_SecondStep_PrePOG *
                                      DQMMessageLoggerClient )

DQMOffline_SecondStep_PrePOGMC = cms.Sequence( bTagCollectorSequence )

DQMOffline_SecondStepPOGMC = cms.Sequence( dqmRefHistoRootFileGetter *
                                           DQMOffline_SecondStep_PrePOGMC *
                                           DQMMessageLoggerClient )

