import FWCore.ParameterSet.Config as cms

from CondTools.DQM.DQMReferenceHistogramRootFileEventSetupAnalyzer_cfi import *
from DQMServices.Components.DQMMessageLoggerClient_cff import *
from DQMServices.Components.DQMDcsInfoClient_cfi import *
from DQMServices.Components.DQMFastTimerServiceClient_cfi import *

from DQMOffline.Ecal.ecal_dqm_client_offline_cff import *
from DQM.SiStripMonitorClient.SiStripClientConfig_Tier0_HeavyIons_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff import *
from DQM.DTMonitorClient.dtDQMOfflineClients_cff import *
from DQM.RPCMonitorClient.RPCTier0Client_cff import *
from DQM.CSCMonitorModule.csc_dqm_offlineclient_collisions_cff import *
from DQM.EcalPreshowerMonitorClient.es_dqm_client_offline_cff import *
from DQM.BeamMonitor.AlcaBeamMonitorClient_cff import *
from DQMServices.Components.DQMFEDIntegrityClient_cff import *
from DQM.HcalTasks.OfflineHarvestingSequence_hi import *

DQMOfflineHeavyIons_SecondStep_PreDPG = cms.Sequence( dqmDcsInfoClient *
                                                      ecal_dqm_client_offline *
                                                      SiStripOfflineDQMClientHI *
                                                      PixelOfflineDQMClientWithDataCertificationHI *
													  hcalOfflineHarvesting *
                                                      dtClients *
                                                      rpcTier0Client *
                                                      cscOfflineCollisionsClients *
                                                      es_dqm_client_offline *
                                                      dqmFEDIntegrityClient )

DQMOfflineHeavyIons_SecondStepDPG = cms.Sequence( dqmRefHistoRootFileGetter *
                                         DQMOfflineHeavyIons_SecondStep_PreDPG *
                                         DQMMessageLoggerClientSeq )

from DQMOffline.Muon.muonQualityTests_cff import *
from DQMOffline.EGamma.photonOfflineDQMClient_cff import *
from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_Client_cff import *
from DQMOffline.Trigger.DQMOffline_HLT_Client_cff import *
from DQM.TrackingMonitorClient.TrackingDQMClientHeavyIons_cfi import *

DQMOfflineHeavyIons_SecondStep_PrePOG = cms.Sequence( muonQualityTests 
                                                      * photonOfflineDQMClient
                                                      * triggerOfflineDQMClient 
                                                      * hltOfflineDQMClient
                                                      * alcaBeamMonitorClient
                                                      * hiTrackingDqmClientHeavyIons
                                                      )

DQMOfflineHeavyIons_SecondStepPOG = cms.Sequence( dqmRefHistoRootFileGetter *
                                                  DQMOfflineHeavyIons_SecondStep_PrePOG *
                                                  DQMMessageLoggerClientSeq *
                                                  dqmFastTimerServiceClient)

DQMOfflineHeavyIons_SecondStep = cms.Sequence( dqmRefHistoRootFileGetter *
                                               DQMOfflineHeavyIons_SecondStep_PreDPG *
                                               DQMOfflineHeavyIons_SecondStep_PrePOG *
                                               DQMMessageLoggerClientSeq )
