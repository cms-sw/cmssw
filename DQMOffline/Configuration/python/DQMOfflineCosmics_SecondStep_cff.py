import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLoggerClient_cff import *
from DQMServices.Components.DQMFastTimerServiceClient_cfi import *

from DQMOffline.Ecal.ecal_dqm_client_offline_cosmic_cff import *
from DQM.EcalPreshowerMonitorClient.es_dqm_client_offline_cosmic_cff import *
from DQM.HcalTasks.OfflineHarvestingSequence_cosmic import *
from DQM.SiStripMonitorClient.SiStripClientConfig_Tier0_Cosmic_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff import *
from DQM.DTMonitorClient.dtDQMOfflineClients_Cosmics_cff import *
from DQM.RPCMonitorClient.RPCTier0Client_cff import *
from DQM.CSCMonitorModule.csc_dqm_offlineclient_cosmics_cff import *
from DQMOffline.Muon.gem_dqm_offline_client_cff import *
from DQMServices.Components.DQMFEDIntegrityClient_cff import *

DQMNone = cms.Sequence()

DQMOfflineCosmics_SecondStepEcal = cms.Sequence( ecal_dqm_client_offline *
						es_dqm_client_offline )

DQMOfflineCosmics_SecondStepHcal = cms.Sequence( hcalOfflineHarvesting )

DQMOfflineCosmics_SecondStepTrackerStrip = cms.Sequence( SiStripCosmicDQMClient )

DQMOfflineCosmics_SecondStepTrackerPixel = cms.Sequence( PixelOfflineDQMClientNoDataCertification_cosmics )

DQMOfflineCosmics_SecondStepMuonDPG = cms.Sequence( dtClientsCosmics *
                                                    rpcTier0Client *
                                                    cscOfflineCosmicsClients )

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
_run3_GEM_DQMOfflineCosmics_SecondStepMuonDPG = DQMOfflineCosmics_SecondStepMuonDPG.copy()
_run3_GEM_DQMOfflineCosmics_SecondStepMuonDPG += gemClients
run3_GEM.toReplaceWith(DQMOfflineCosmics_SecondStepMuonDPG, _run3_GEM_DQMOfflineCosmics_SecondStepMuonDPG)

DQMOfflineCosmics_SecondStepFED = cms.Sequence( dqmFEDIntegrityClient )

DQMOfflineCosmics_SecondStep_PreDPG = cms.Sequence( 
                                                    DQMOfflineCosmics_SecondStepEcal *
                                                    DQMOfflineCosmics_SecondStepHcal *
                                                    DQMOfflineCosmics_SecondStepTrackerStrip *
                                                    DQMOfflineCosmics_SecondStepTrackerPixel *
                                                    DQMOfflineCosmics_SecondStepMuonDPG *
                                                    DQMOfflineCosmics_SecondStepFED )


DQMOfflineCosmics_SecondStepDPG = cms.Sequence(
                                                DQMOfflineCosmics_SecondStep_PreDPG *
                                                DQMMessageLoggerClientSeq )

from DQM.TrackingMonitorClient.TrackingClientConfig_Tier0_Cosmic_cff import *
from DQMOffline.Muon.muonQualityTests_cff import *
from DQMOffline.EGamma.photonOfflineDQMClient_cff import *
from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_Client_cff import *
from DQMOffline.Trigger.DQMOffline_HLT_Client_cff import *
from DQMOffline.JetMET.SusyPostProcessor_cff import *

DQMOfflineCosmics_SecondStepTracking = cms.Sequence( TrackingCosmicDQMClient )

DQMOfflineCosmics_SecondStepMUO = cms.Sequence( cosmicMuonQualityTests )

DQMOfflineCosmics_SecondStepEGamma = cms.Sequence( photonOfflineDQMClient )

DQMOfflineCosmics_SecondStepL1T = cms.Sequence( l1TriggerDqmOfflineCosmicsClient )

DQMOfflineCosmics_SecondStepTrigger = cms.Sequence( triggerOfflineDQMClient *
							hltOfflineDQMClient )

DQMOfflineCosmics_SecondStepJetMET = cms.Sequence( SusyPostProcessorSequence )
DQMOfflineCosmics_SecondStep_PrePOG = cms.Sequence( DQMOfflineCosmics_SecondStepTracking *
                                                    DQMOfflineCosmics_SecondStepMUO *
                                                    DQMOfflineCosmics_SecondStepEGamma *
                                                    DQMOfflineCosmics_SecondStepL1T *
                                                    DQMOfflineCosmics_SecondStepJetMET 
                                                    )

DQMOfflineCosmics_SecondStep_PrePOG.remove(fsqClient)
DQMOfflineCosmics_SecondStepPOG = cms.Sequence(
                                                DQMOfflineCosmics_SecondStep_PrePOG *
                                                DQMMessageLoggerClientSeq *
                                                dqmFastTimerServiceClient)

DQMOfflineCosmics_SecondStep = cms.Sequence( 
                                             DQMOfflineCosmics_SecondStep_PreDPG *
                                             DQMOfflineCosmics_SecondStep_PrePOG *
					     DQMOfflineCosmics_SecondStepTrigger *
                                             DQMMessageLoggerClientSeq )

DQMOfflineCosmics_SecondStep_FakeHLT = cms.Sequence(DQMOfflineCosmics_SecondStep ) 
DQMOfflineCosmics_SecondStep_FakeHLT.remove( DQMOfflineCosmics_SecondStepTrigger )

