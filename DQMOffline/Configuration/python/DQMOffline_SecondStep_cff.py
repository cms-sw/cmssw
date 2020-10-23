import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLoggerClient_cff import *
from DQMServices.Components.DQMFastTimerServiceClient_cfi import *

from DQMOffline.Ecal.ecal_dqm_client_offline_cff import *
from DQM.EcalPreshowerMonitorClient.es_dqm_client_offline_cff import *
from DQM.SiStripMonitorClient.SiStripClientConfig_Tier0_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff import *
from DQM.DTMonitorClient.dtDQMOfflineClients_cff import *
from DQM.RPCMonitorClient.RPCTier0Client_cff import *
from DQM.CSCMonitorModule.csc_dqm_offlineclient_collisions_cff import *
from DQMOffline.Hcal.HcalDQMOfflinePostProcessor_cff import *
from DQM.HcalTasks.OfflineHarvestingSequence_pp import *
from DQMServices.Components.DQMFEDIntegrityClient_cff import *
from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *
from DQM.SiTrackerPhase2.Phase2TrackerDQMHarvesting_cff import *

DQMNone = cms.Sequence()

DQMOffline_SecondStepEcal = cms.Sequence( ecal_dqm_client_offline *
					  es_dqm_client_offline )

DQMOffline_SecondStepTrackerStrip = cms.Sequence( SiStripOfflineDQMClient )

DQMOffline_SecondStepTrackerPixel = cms.Sequence( PixelOfflineDQMClientNoDataCertification )

DQMOffline_SecondStepMuonDPG = cms.Sequence( dtClients *
                                             rpcTier0Client *
                                             cscOfflineCollisionsClients )

DQMOffline_SecondStepHcal = cms.Sequence( hcalOfflineHarvesting )

DQMOffline_SecondStepHcal2 = cms.Sequence(  HcalDQMOfflinePostProcessor )

DQMOffline_SecondStepFED = cms.Sequence( dqmFEDIntegrityClient )

DQMOffline_SecondStepL1T = cms.Sequence( l1TriggerDqmOfflineClient )

DQMOffline_SecondStep_PreDPG = cms.Sequence( 
                                             DQMOffline_SecondStepEcal *
                                             DQMOffline_SecondStepTrackerStrip *
					     DQMOffline_SecondStepTrackerPixel *
                                             DQMOffline_SecondStepMuonDPG *
					     DQMOffline_SecondStepHcal *
					     DQMOffline_SecondStepHcal2 *
                                             DQMOffline_SecondStepFED *
					     DQMOffline_SecondStepL1T )

DQMOffline_SecondStepDPG = cms.Sequence(
                                         DQMOffline_SecondStep_PreDPG *
                                         DQMMessageLoggerClientSeq )


from DQM.TrackingMonitorClient.TrackingClientConfig_Tier0_cff import *
from DQMOffline.Muon.muonQualityTests_cff import *
from DQMOffline.EGamma.egammaPostProcessing_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_Client_cff import *
from DQMOffline.Trigger.DQMOffline_HLT_Client_cff import *
from DQMOffline.RecoB.dqmCollector_cff import *
from DQM.BeamMonitor.AlcaBeamMonitorClient_cff import *
from DQMOffline.JetMET.SusyPostProcessor_cff import *

DQMOffline_SecondStepTracking = cms.Sequence ( TrackingOfflineDQMClient )

DQMOffline_SecondStepMUO = cms.Sequence ( muonQualityTests )

DQMOffline_SecondStepEGamma = cms.Sequence( egammaPostProcessing )

DQMOffline_SecondStepTrigger = cms.Sequence( triggerOfflineDQMClient *
						hltOfflineDQMClient )

DQMOffline_SecondStepBTag = cms.Sequence( bTagCollectorSequenceDATA )

DQMOffline_SecondStepBeam = cms.Sequence( alcaBeamMonitorClient )

DQMOffline_SecondStepJetMET = cms.Sequence( SusyPostProcessorSequence )

DQMOffline_SecondStep_PrePOG = cms.Sequence( DQMOffline_SecondStepTracking *
                                             DQMOffline_SecondStepMUO *
                                             DQMOffline_SecondStepEGamma *
                                             DQMOffline_SecondStepTrigger *
                                             DQMOffline_SecondStepBTag *
                                             DQMOffline_SecondStepBeam *
                                             DQMOffline_SecondStepJetMET )

DQMOffline_SecondStepPOG = cms.Sequence(
                                         DQMOffline_SecondStep_PrePOG *
                                         DQMMessageLoggerClientSeq )


HLTMonitoringClient = cms.Sequence(trackingMonitorClientHLT * trackingForDisplacedJetMonitorClientHLT)
HLTMonitoringClientPA= cms.Sequence(trackingMonitorClientHLT * PAtrackingMonitorClientHLT)

DQMOffline_SecondStep = cms.Sequence(
                                      DQMOffline_SecondStep_PreDPG *
                                      DQMOffline_SecondStep_PrePOG *
                                      HLTMonitoringClient *
                                      DQMMessageLoggerClientSeq *
                                      dqmFastTimerServiceClient)

DQMOffline_SecondStep_ExtraHLT = cms.Sequence( hltOfflineDQMClientExtra )

DQMOffline_SecondStep_FakeHLT = cms.Sequence( DQMOffline_SecondStep )
DQMOffline_SecondStep_FakeHLT.remove( HLTMonitoringClient )
DQMOffline_SecondStep_FakeHLT.remove( DQMOffline_SecondStepTrigger )

DQMOffline_SecondStep_PrePOGMC = cms.Sequence( bTagCollectorSequenceDATA )

DQMOffline_SecondStepPOGMC = cms.Sequence( DQMOffline_SecondStep_PrePOGMC *
                                           DQMMessageLoggerClientSeq )

# Harvest
from DQMOffline.JetMET.dataCertificationJetMET_cff import *
from DQM.SiOuterTracker.OuterTrackerClientConfig_cff import *
from DQM.CTPPS.ctppsDQM_cff import *
from Validation.RecoTau.DQMSequences_cfi import *
from DQM.TrackingMonitorClient.pixelTrackingEffFromHitPattern_cff import *

DQMHarvestTrackerStrip = cms.Sequence ( SiStripOfflineDQMClient )

DQMHarvestTrackerPixel = cms.Sequence ( PixelOfflineDQMClientNoDataCertification )

DQMHarvestTrack = cms.Sequence ( TrackingOfflineDQMClient )

DQMHarvestTrigger = cms.Sequence ( triggerOfflineDQMClient *
				    hltOfflineDQMClient )

DQMHarvestFED = cms.Sequence ( dqmFEDIntegrityClient )

DQMHarvestBeam = cms.Sequence ( alcaBeamMonitorClient )

DQMHarvestTAU = cms.Sequence ( runTauEff )

DQMHarvestL1T = cms.Sequence( l1TriggerDqmOfflineClient )

DQMHarvestL1TEgamma = cms.Sequence( l1TriggerEgDqmOfflineClient )

DQMHarvestL1TMuon = cms.Sequence( l1TriggerMuonDqmOfflineClient )

DQMHarvestCommon = cms.Sequence( DQMMessageLoggerClientSeq *
                                 DQMHarvestTrackerStrip *
                                 DQMHarvestTrack *
                                 DQMHarvestTrackerPixel *
				 DQMHarvestTrigger *
                                 DQMHarvestFED *
                                 DQMHarvestBeam *
                                 DQMHarvestTAU *
                                 dqmFastTimerServiceClient
                                )

DQMHarvestCommonFakeHLT = cms.Sequence( DQMHarvestCommon )
DQMHarvestCommonFakeHLT.remove( DQMHarvestTrigger )

DQMHarvestCommonSiStripZeroBias = cms.Sequence(
                                               DQMMessageLoggerClientSeq *
                                               DQMHarvestTrackerStrip *
                                               DQMHarvestTrack *
                                               DQMHarvestTrackerPixel *
                                               DQMHarvestTrigger *
                                               DQMHarvestL1T *
                                               DQMHarvestFED *
                                               DQMHarvestBeam *
                                               dqmFastTimerServiceClient
                                               )

DQMHarvestCommonSiStripZeroBiasFakeHLT = cms.Sequence( DQMHarvestCommonSiStripZeroBias )
DQMHarvestCommonSiStripZeroBiasFakeHLT.remove( DQMHarvestTrigger )

DQMHarvestTracking = cms.Sequence( TrackingOfflineDQMClient *
                                   dqmFastTimerServiceClient )

DQMHarvestTrackingZeroBias = cms.Sequence( TrackingOfflineDQMClientZeroBias *
                                           dqmFastTimerServiceClient )

DQMHarvestPixelTracking = cms.Sequence( pixelTrackingEffFromHitPattern )

DQMHarvestOuterTracker = cms.Sequence(
                                 OuterTrackerClient *
                                 dqmFEDIntegrityClient *
                                 DQMMessageLoggerClientSeq *
                                 dqmFastTimerServiceClient
                                 )
DQMHarvestTrackerPhase2 = cms.Sequence(trackerphase2DQMHarvesting)


DQMHarvestCTPPS = cms.Sequence( ctppsDQMOfflineHarvest )

DQMHarvestMuon = cms.Sequence( dtClients *
                               rpcTier0Client *
                               cscOfflineCollisionsClients *
                               muonQualityTests
                               )

DQMHarvestEcal = cms.Sequence( ecal_dqm_client_offline *
                                es_dqm_client_offline
                              )

DQMHarvestHcal = cms.Sequence( hcalOfflineHarvesting )

DQMHarvestHcal2 = cms.Sequence( HcalDQMOfflinePostProcessor )

DQMHarvestJetMET = cms.Sequence( SusyPostProcessorSequence )

DQMHarvestEGamma = cms.Sequence( egammaPostProcessing )

DQMHarvestBTag = cms.Sequence( bTagCollectorSequenceDATA )

from PhysicsTools.NanoAOD.nanoDQM_cff import *
from Validation.RecoParticleFlow.DQMForPF_MiniAOD_cff import *

DQMHarvestMiniAOD = cms.Sequence( dataCertificationJetMETSequence * muonQualityTests_miniAOD * DQMHarvestPF )
DQMHarvestNanoAOD = cms.Sequence( nanoHarvest )

