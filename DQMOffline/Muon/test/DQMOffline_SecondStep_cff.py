#### Modifications needed to run only Offline Muon DQM (by A. Calderon) ####


import FWCore.ParameterSet.Config as cms

from CondTools.DQM.DQMReferenceHistogramRootFileEventSetupAnalyzer_cfi import *
from DQMServices.Components.DQMMessageLoggerClient_cff import *

from DQMOffline.Ecal.ecal_dqm_client_offline_cff import *
from DQM.HcalMonitorModule.hcal_dqm_client_fileT0_cff import *
from DQM.SiStripMonitorClient.SiStripClientConfig_Tier0_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff import *
from DQM.DTMonitorClient.dtDQMOfflineClients_cff import *
from DQM.RPCMonitorClient.RPCTier0Client_cff import *
from DQM.CSCMonitorModule.csc_dqm_offlineclient_collisions_cff import *
from DQM.EcalPreshowerMonitorClient.es_dqm_client_offline_cff import *
from DQM.BeamMonitor.AlcaBeamMonitorClient_cff import *
from DQMServices.Components.DQMFEDIntegrityClient_cff import *


DQMOffline_SecondStep_PreDPG = cms.Sequence( 
                                             #ecal_dqm_client_offline *
                                             #hcalOfflineDQMClient *
                                             SiStripOfflineDQMClient *
                                             PixelOfflineDQMClientNoDataCertification *
                                             dtClients *
                                             rpcTier0Client *
                                             cscOfflineCollisionsClients )#*
                                             #es_dqm_client_offline *
                                            #dqmFEDIntegrityClient
                                             

DQMOffline_SecondStepDPG = cms.Sequence( #dqmRefHistoRootFileGetter *
                                         DQMOffline_SecondStep_PreDPG *
                                         DQMMessageLoggerClientSeq )


from DQMOffline.Muon.muonQualityTests_cff import *
from DQMOffline.EGamma.egammaPostProcessing_cff import *
from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_Client_cff import *
from DQMOffline.Trigger.DQMOffline_HLT_Client_cff import *
from DQMOffline.RecoB.dqmCollector_cff import *
from DQMOffline.JetMET.SusyPostProcessor_cff import *

#DQMOffline_SecondStep_PrePOG = cms.Sequence( muonQualityTests *
#                                             egammaPostProcessing *
#                                             l1TriggerDqmOfflineClient *
#                                             triggerOfflineDQMClient *
#                                             hltOfflineDQMClient *
#                                             bTagCollectorSequence *
#                                             alcaBeamMonitorClient *
#                                             SusyPostProcessorSequence )

DQMOffline_SecondStep_PrePOG = cms.Sequence( muonQualityTests )

DQMOffline_SecondStepPOG = cms.Sequence( dqmRefHistoRootFileGetter *
                                         DQMOffline_SecondStep_PrePOG *
                                         DQMMessageLoggerClientSeq )

DQMOffline_SecondStep = cms.Sequence( dqmRefHistoRootFileGetter *
                                      DQMOffline_SecondStep_PreDPG *
                                      DQMOffline_SecondStep_PrePOG *
                                      DQMMessageLoggerClientSeq )

DQMOffline_SecondStep_PrePOGMC = cms.Sequence( bTagCollectorSequence )

DQMOffline_SecondStepPOGMC = cms.Sequence( dqmRefHistoRootFileGetter *
                                           DQMOffline_SecondStep_PrePOGMC *
                                           DQMMessageLoggerClientSeq )


#DQMHarvestCommon = cms.Sequence(
#                                 SiStripOfflineDQMClient *
#                                 PixelOfflineDQMClientNoDataCertification *
#                                 l1TriggerDqmOfflineClient *
#                                 triggerOfflineDQMClient *
#                                 hltOfflineDQMClient *
#                                 dqmFEDIntegrityClient *
#                                 alcaBeamMonitorClient *
#                                 SusyPostProcessorSequence
#                                )

DQMHarvestCommon = cms.Sequence(
                                 SiStripOfflineDQMClient *
                                 PixelOfflineDQMClientNoDataCertification *
                                # l1TriggerDqmOfflineClient *
                                # triggerOfflineDQMClient *
                                # hltOfflineDQMClient *
                                 dqmFEDIntegrityClient 
                                # alcaBeamMonitorClient *
                                # SusyPostProcessorSequence
                                )


DQMHarvestMuon = cms.Sequence( #dtClients *
                               # rpcTier0Client *
                               # cscOfflineCollisionsClients *
                                muonQualityTests
                              )

DQMHarvestEcal = cms.Sequence( ecal_dqm_client_offline *
                                es_dqm_client_offline
                              )
DQMHarvestHcal = cms.Sequence( hcalOfflineDQMClient )

DQMHarvestJetMET = cms.Sequence( SusyPostProcessorSequence )
                                             
DQMStepTwo_Common = cms.Sequence( DQMHarvestCommon )

DQMStepTwo_Common_Muon = cms.Sequence( DQMHarvestCommon * DQMHarvestMuon)

DQMStepTwo_Common_Hcal_JetMET = cms.Sequence( DQMHarvestCommon * DQMHarvestHcal * DQMHarvestJetMET)

DQMStepTwo_Common_Ecal = cms.Sequence( DQMHarvestCommon * DQMHarvestEcal)

DQMStepTwo_Common_Ecal_Hcal = cms.Sequence( DQMHarvestCommon * DQMHarvestEcal * DQMHarvestHcal)
                                   
DQMStepTwo_Common_Muon_JetMET = cms.Sequence( DQMHarvestCommon * DQMHarvestMuon * DQMHarvestJetMET)
