import FWCore.ParameterSet.Config as cms

from CondTools.DQM.DQMReferenceHistogramRootFileEventSetupAnalyzer_cfi import *
from DQMServices.Components.DQMMessageLoggerClient_cff import *
from DQMServices.Components.DQMDcsInfoClient_cfi import *
from DQMServices.Components.DQMFastTimerServiceClient_cfi import *

from DQMOffline.Ecal.ecal_dqm_client_offline_cff import *
from DQM.SiStripMonitorClient.SiStripClientConfig_Tier0_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff import *
from DQM.DTMonitorClient.dtDQMOfflineClients_cff import *
from DQM.RPCMonitorClient.RPCTier0Client_cff import *
from DQM.CSCMonitorModule.csc_dqm_offlineclient_collisions_cff import *
from DQM.EcalPreshowerMonitorClient.es_dqm_client_offline_cff import *
from DQM.BeamMonitor.AlcaBeamMonitorClient_cff import *
from DQMServices.Components.DQMFEDIntegrityClient_cff import *
from Validation.RecoTau.DQMSequences_cfi import *
from DQMOffline.Hcal.HcalDQMOfflinePostProcessor_cff import *
from DQM.HcalTasks.OfflineHarvestingSequence_pp import *

DQMOffline_SecondStep_PreDPG = cms.Sequence( dqmDcsInfoClient *
                                             ecal_dqm_client_offline *
                                             SiStripOfflineDQMClient *
                                             PixelOfflineDQMClientNoDataCertification *
                                             dtClients *
                                             rpcTier0Client *
                                             cscOfflineCollisionsClients *
                                             es_dqm_client_offline *
                                             #hcalOfflineHarvesting *
                                             HcalDQMOfflinePostProcessor * 
                                             dqmFEDIntegrityClient )

DQMOffline_SecondStepDPG = cms.Sequence( dqmRefHistoRootFileGetter *
                                         DQMOffline_SecondStep_PreDPG *
                                         DQMMessageLoggerClientSeq )

from DQMOffline.Muon.muonQualityTests_cff import *
from DQMOffline.EGamma.egammaPostProcessing_cff import *
from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_Client_cff import *
from DQMOffline.Trigger.DQMOffline_HLT_Client_cff import *
from DQMOffline.RecoB.dqmCollector_cff import *
from DQMOffline.JetMET.SusyPostProcessor_cff import *
from DQMOffline.JetMET.dataCertificationJetMET_cff import *
from DQM.TrackingMonitorClient.TrackingClientConfig_Tier0_cff import *
from DQM.Phase2OuterTracker.OuterTrackerClientConfig_cff import *

DQMOffline_SecondStep_PrePOG = cms.Sequence( TrackingOfflineDQMClient *
                                             muonQualityTests *
                                             egammaPostProcessing *
                                             triggerOfflineDQMClient *
                                             hltOfflineDQMClient *
                                             bTagCollectorSequenceDATA *
                                             alcaBeamMonitorClient *
                                             SusyPostProcessorSequence *
                                             runTauEff)
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel

DQMOffline_SecondStepPOG = cms.Sequence( dqmRefHistoRootFileGetter *
                                         DQMOffline_SecondStep_PrePOG *
                                         DQMMessageLoggerClientSeq )


HLTMonitoringClient = cms.Sequence(trackingMonitorClientHLT)
HLTMonitoringClientPA= cms.Sequence(trackingMonitorClientHLT * PAtrackingMonitorClientHLT)
DQMOffline_SecondStep = cms.Sequence( dqmRefHistoRootFileGetter *
                                      DQMOffline_SecondStep_PreDPG *
                                      DQMOffline_SecondStep_PrePOG *
                                      HLTMonitoringClient *
                                      DQMMessageLoggerClientSeq *
                                      dqmFastTimerServiceClient)
DQMOffline_SecondStep_FakeHLT = cms.Sequence( DQMOffline_SecondStep )
DQMOffline_SecondStep_FakeHLT.remove( HLTMonitoringClient )

DQMOffline_SecondStep_PrePOGMC = cms.Sequence( bTagCollectorSequenceDATA )

DQMOffline_SecondStepPOGMC = cms.Sequence( dqmRefHistoRootFileGetter *
                                           DQMOffline_SecondStep_PrePOGMC *
                                           DQMMessageLoggerClientSeq )

from DQMOffline.L1Trigger.L1TriggerDqmOffline_SecondStep_cff import *

DQMHarvestCommon = cms.Sequence( dqmRefHistoRootFileGetter *
                                 DQMMessageLoggerClientSeq *
                                 dqmDcsInfoClient *
                                 SiStripOfflineDQMClient *
                                 TrackingOfflineDQMClient *
                                 PixelOfflineDQMClientNoDataCertification *
                                 triggerOfflineDQMClient *
                                 hltOfflineDQMClient *
                                 dqmFEDIntegrityClient *
                                 alcaBeamMonitorClient *
                                 runTauEff *
                                 dqmFastTimerServiceClient *
                                 DQMHarvestL1Trigger
                                )
DQMHarvestCommonSiStripZeroBias = cms.Sequence(dqmRefHistoRootFileGetter *
                                               DQMMessageLoggerClientSeq *
                                               dqmDcsInfoClient *
                                               SiStripOfflineDQMClient *
                                               TrackingOfflineDQMClient *
                                               PixelOfflineDQMClientNoDataCertification *
                                               triggerOfflineDQMClient *
                                               hltOfflineDQMClient *
                                               dqmFEDIntegrityClient *
                                               alcaBeamMonitorClient *
                                               runTauEff  *
                                               dqmFastTimerServiceClient
                                               )

DQMHarvestTracking = cms.Sequence( TrackingOfflineDQMClient *
                                   dqmFastTimerServiceClient )

DQMHarvestOuterTracker = cms.Sequence( dqmRefHistoRootFileGetter *
                                 dqmDcsInfoClient *
                                 OuterTrackerClient *
                                 dqmFEDIntegrityClient *
                                 DQMMessageLoggerClientSeq *
                                 dqmFastTimerServiceClient
                                 )

DQMHarvestMuon = cms.Sequence( dtClients *
                               rpcTier0Client *
                               cscOfflineCollisionsClients *
                               muonQualityTests
                               )

DQMHarvestEcal = cms.Sequence( ecal_dqm_client_offline *
                                es_dqm_client_offline
                              )

#DQMHarvestHcal = cms.Sequence(hcalOfflineHarvesting)
DQMHarvestHcal = cms.Sequence()

DQMHarvestJetMET = cms.Sequence( SusyPostProcessorSequence )

DQMHarvestEGamma = cms.Sequence( egammaPostProcessing )

DQMHarvestBTag = cms.Sequence( bTagCollectorSequenceDATA )

DQMHarvestMiniAOD = cms.Sequence( dataCertificationJetMETSequence * muonQualityTests_miniAOD)
