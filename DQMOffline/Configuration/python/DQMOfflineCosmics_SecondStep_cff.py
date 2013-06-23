import FWCore.ParameterSet.Config as cms

from CondTools.DQM.DQMReferenceHistogramRootFileEventSetupAnalyzer_cfi import *
from DQMServices.Components.DQMMessageLoggerClient_cff import *
from DQMServices.Components.DQMDcsInfoClient_cfi import *

from DQMOffline.Ecal.ecal_dqm_client_offline_cosmic_cff import *
from DQM.HcalMonitorModule.hcal_dqm_client_fileT0_cff import *
from DQM.SiStripMonitorClient.SiStripClientConfig_Tier0_Cosmic_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff import *
from DQM.DTMonitorClient.dtDQMOfflineClients_Cosmics_cff import *
from DQM.RPCMonitorClient.RPCTier0Client_cff import *
from DQM.CSCMonitorModule.csc_dqm_offlineclient_cosmics_cff import *
from DQM.EcalPreshowerMonitorClient.es_dqm_client_offline_cosmic_cff import *
from DQMServices.Components.DQMFEDIntegrityClient_cff import *

DQMOfflineCosmics_SecondStep_PreDPG = cms.Sequence( dqmDcsInfoClient * 
                                                    ecal_dqm_client_offline *
                                                    hcalOfflineDQMClient *
                                                    SiStripCosmicDQMClient *
                                                    PixelOfflineDQMClientNoDataCertification *
                                                    dtClientsCosmics *
                                                    rpcTier0Client *
                                                    cscOfflineCosmicsClients *
                                                    es_dqm_client_offline *
                                                    dqmFEDIntegrityClient )


DQMOfflineCosmics_SecondStepDPG = cms.Sequence( dqmRefHistoRootFileGetter *
                                                DQMOfflineCosmics_SecondStep_PreDPG *
                                                DQMMessageLoggerClientSeq )

from DQMOffline.Muon.muonQualityTests_cff import *
from DQMOffline.EGamma.photonOfflineDQMClient_cff import *
from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_Client_cff import *
from DQMOffline.Trigger.DQMOffline_HLT_Client_cff import *
from DQMOffline.JetMET.SusyPostProcessor_cff import *

DQMOfflineCosmics_SecondStep_PrePOG = cms.Sequence( cosmicMuonQualityTests *
                                                    photonOfflineDQMClient *
#                                                    l1TriggerDqmOfflineClient * 
                                                    triggerOfflineDQMClient *
                                                    hltOfflineDQMClient *
                                                    SusyPostProcessorSequence )
 
DQMOfflineCosmics_SecondStepPOG = cms.Sequence( dqmRefHistoRootFileGetter *
                                                DQMOfflineCosmics_SecondStep_PrePOG *
                                                DQMMessageLoggerClientSeq )

DQMOfflineCosmics_SecondStep = cms.Sequence( dqmRefHistoRootFileGetter *
                                             DQMOfflineCosmics_SecondStep_PreDPG *
                                             DQMOfflineCosmics_SecondStep_PrePOG *
                                             DQMMessageLoggerClientSeq )

