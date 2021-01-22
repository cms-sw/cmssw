import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLoggerClient_cff import *
from DQMServices.Components.DQMFastTimerServiceClient_cfi import *

from DQMOffline.Ecal.ecal_dqm_client_offline_cff import *
from DQM.EcalPreshowerMonitorClient.es_dqm_client_offline_cff import *
from DQM.SiStripMonitorClient.SiStripClientConfig_Tier0_HeavyIons_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff import *
from DQM.HcalTasks.OfflineHarvestingSequence_hi import *
from DQM.DTMonitorClient.dtDQMOfflineClients_cff import *
from DQM.RPCMonitorClient.RPCTier0Client_cff import *
from DQM.CSCMonitorModule.csc_dqm_offlineclient_collisions_cff import *
from DQMOffline.Muon.gem_dqm_offline_client_cff import *
from DQMServices.Components.DQMFEDIntegrityClient_cff import *

DQMNone = cms.Sequence()

DQMOfflineHeavyIons_SecondStepEcal = cms.Sequence( ecal_dqm_client_offline *
						    es_dqm_client_offline )

DQMOfflineHeavyIons_SecondStepTrackerStrip = cms.Sequence( SiStripOfflineDQMClientHI )

DQMOfflineHeavyIons_SecondStepTrackerPixel = cms.Sequence( PixelOfflineDQMClientWithDataCertificationHI )

DQMOfflineHeavyIons_SecondStepHcal = cms.Sequence( hcalOfflineHarvesting )

DQMOfflineHeavyIons_SecondStepMuonDPG = cms.Sequence(  dtClients *
                                                      rpcTier0Client *
                                                      cscOfflineCollisionsClients )

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
_run3_GEM_DQMOfflineHeavyIons_SecondStepMuonDPG = DQMOfflineHeavyIons_SecondStepMuonDPG.copy()
_run3_GEM_DQMOfflineHeavyIons_SecondStepMuonDPG += gemClients
run3_GEM.toReplaceWith(DQMOfflineHeavyIons_SecondStepMuonDPG, _run3_GEM_DQMOfflineHeavyIons_SecondStepMuonDPG)

DQMOfflineHeavyIons_SecondStepFED = cms.Sequence( dqmFEDIntegrityClient )

DQMOfflineHeavyIons_SecondStep_PreDPG = cms.Sequence( 
						      DQMOfflineHeavyIons_SecondStepEcal *
                                                      DQMOfflineHeavyIons_SecondStepTrackerStrip *
                                                      DQMOfflineHeavyIons_SecondStepTrackerPixel *
                                                      DQMOfflineHeavyIons_SecondStepHcal *
                                                      DQMOfflineHeavyIons_SecondStepMuonDPG *
                                                      DQMOfflineHeavyIons_SecondStepFED 
							)

DQMOfflineHeavyIons_SecondStepDPG = cms.Sequence(
                                         DQMOfflineHeavyIons_SecondStep_PreDPG *
                                         DQMMessageLoggerClientSeq )

from DQMOffline.Muon.muonQualityTests_cff import *
from DQMOffline.EGamma.photonOfflineDQMClient_cff import *
from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_Client_cff import *
from DQMOffline.Trigger.DQMOffline_HLT_Client_cff import *
from DQM.TrackingMonitorClient.TrackingDQMClientHeavyIons_cfi import *
from DQM.BeamMonitor.AlcaBeamMonitorClient_cff import *

DQMOfflineHeavyIons_SecondStepMUO = cms.Sequence( muonQualityTests )

DQMOfflineHeavyIons_SecondStepEGamma = cms.Sequence( photonOfflineDQMClient )

DQMOfflineHeavyIons_SecondStepTrigger  = cms.Sequence( triggerOfflineDQMClient *
							hltOfflineDQMClient )
DQMOfflineHeavyIons_SecondStepBeam = cms.Sequence( alcaBeamMonitorClient )

DQMOfflineHeavyIons_SecondStepTracking = cms.Sequence( hiTrackingDqmClientHeavyIons )

DQMOfflineHeavyIons_SecondStep_PrePOG = cms.Sequence( DQMOfflineHeavyIons_SecondStepMUO * 
                                                      DQMOfflineHeavyIons_SecondStepEGamma *
                                                      DQMOfflineHeavyIons_SecondStepTrigger *
                                                      DQMOfflineHeavyIons_SecondStepBeam *
						      DQMOfflineHeavyIons_SecondStepTracking
                                                      )

DQMOfflineHeavyIons_SecondStepPOG = cms.Sequence(
                                                  DQMOfflineHeavyIons_SecondStep_PrePOG *
                                                  DQMMessageLoggerClientSeq *
                                                  dqmFastTimerServiceClient)

DQMOfflineHeavyIons_SecondStep = cms.Sequence(
                                               DQMOfflineHeavyIons_SecondStep_PreDPG *
                                               DQMOfflineHeavyIons_SecondStep_PrePOG *
                                               DQMMessageLoggerClientSeq )

DQMOfflineHeavyIons_SecondStep_FakeHLT = cms.Sequence( DQMOfflineHeavyIons_SecondStep )
DQMOfflineHeavyIons_SecondStep_FakeHLT.remove( DQMOfflineHeavyIons_SecondStepTrigger )

