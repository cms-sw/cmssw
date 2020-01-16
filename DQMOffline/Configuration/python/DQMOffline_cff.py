import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLogger_cfi import *
from DQMServices.Components.DQMDcsInfo_cfi import *
from DQMServices.Components.DQMFastTimerService_cff import *

from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *
from DQMOffline.Ecal.ecal_dqm_source_offline_cff import *
from DQM.EcalPreshowerMonitorModule.es_dqm_source_offline_cff import *
from DQM.HcalTasks.OfflineSourceSequence_pp import *
from DQMOffline.Hcal.HcalDQMOfflineSequence_cff import *
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_source_cff import *
from DQM.DTMonitorModule.dtDQMOfflineSources_cff import *
from DQM.RPCMonitorClient.RPCTier0Source_cff import *
from DQM.CSCMonitorModule.csc_dqm_sourceclient_offline_cff import *
from DQM.CastorMonitor.castor_dqm_sourceclient_offline_cff import *
from DQM.CTPPS.ctppsDQM_cff import *
from DQM.SiTrackerPhase2.Phase2TrackerDQMFirstStep_cff import *

DQMNone = cms.Sequence()

DQMMessageLoggerSeq = cms.Sequence( DQMMessageLogger )

DQMOfflineDCS = cms.Sequence( dqmDcsInfo )

# L1 trigger sequences
DQMOfflineL1T = cms.Sequence( l1TriggerDqmOffline ) # L1 emulator is run within this sequence for real data

DQMOfflineL1TEgamma = cms.Sequence( l1TriggerEgDqmOffline )

DQMOfflineL1TMuon = cms.Sequence( l1TriggerMuonDqmOffline )

#DPGs
DQMOfflineEcal = cms.Sequence( ecal_dqm_source_offline *
				es_dqm_source_offline )

DQMOfflineHcal = cms.Sequence( hcalOfflineSourceSequence )

DQMOfflineHcal2 = cms.Sequence( HcalDQMOfflineSequence )

DQMOfflineTrackerStrip = cms.Sequence( SiStripDQMTier0 )

DQMOfflineTrackerPixel = cms.Sequence( 	siPixelOfflineDQM_source )

DQMOfflineMuonDPG = cms.Sequence( dtSources *
                                  rpcTier0Source *
                                  cscSources )

DQMOfflineCASTOR = cms.Sequence( castorSources )

DQMOfflineCTPPS = cms.Sequence( ctppsDQMOfflineSource )

DQMOfflinePreDPG = cms.Sequence( DQMOfflineDCS *
				 DQMOfflineL1T *
                                 DQMOfflineEcal *
                                 DQMOfflineHcal *
				 DQMOfflineHcal2 *
                                 DQMOfflineTrackerStrip *
				 DQMOfflineTrackerPixel *
				 DQMOfflineMuonDPG *
                                 DQMOfflineCASTOR *
                                 DQMOfflineCTPPS )

DQMOfflineDPG = cms.Sequence( DQMOfflinePreDPG *
                              DQMMessageLogger )

from DQM.TrackingMonitorSource.TrackingSourceConfig_Tier0_cff import *
from DQMOffline.RecoB.PrimaryVertexMonitor_cff import *
from DQM.TrackingMonitor.trackingRecoMaterialAnalyzer_cfi import materialDumperAnalyzer
from DQMOffline.Muon.muonMonitors_cff import *
from DQMOffline.JetMET.jetMETDQMOfflineSource_cff import *
from DQMOffline.EGamma.egammaDQMOffline_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_cff import *
from DQMOffline.RecoB.dqmAnalyzer_cff import *
from DQM.BeamMonitor.AlcaBeamMonitor_cff import *
from DQM.Physics.DQMPhysics_cff import *

DQMOfflineVertex = cms.Sequence( pvMonitor )

materialDumperAnalyzer.usePV = True
DQMOfflineTracking = cms.Sequence( TrackingDQMSourceTier0 *
                                   DQMOfflineVertex *
                                   materialDumperAnalyzer )

DQMOfflineMUO = cms.Sequence(muonMonitors)
muonRecoAnalyzer.doMVA =         cms.bool( True )
muonRecoAnalyzer_miniAOD.doMVA = cms.bool( True )

DQMOfflineJetMET = cms.Sequence( jetMETDQMOfflineSource )

DQMOfflineEGamma = cms.Sequence( egammaDQMOffline )

DQMOfflineTrigger = cms.Sequence( triggerOfflineDQMSource )

DQMOfflineBTag = cms.Sequence( bTagPlotsDATA )

DQMOfflineBeam = cms.Sequence( alcaBeamMonitor )

DQMOfflinePhysics = cms.Sequence( dqmPhysics )

DQMOfflinePrePOG = cms.Sequence( DQMOfflineTracking *
                                 DQMOfflineMUO *
                                 DQMOfflineJetMET *
                                 DQMOfflineEGamma *
                                 DQMOfflineTrigger *
                                 DQMOfflineBTag *
                                 DQMOfflineBeam *
                                 DQMOfflinePhysics )


DQMOfflinePOG = cms.Sequence( DQMOfflinePrePOG *
                              DQMMessageLogger )

HLTMonitoring = cms.Sequence( OfflineHLTMonitoring )
HLTMonitoringPA = cms.Sequence( OfflineHLTMonitoringPA )

# Data
DQMOffline = cms.Sequence( DQMOfflinePreDPG *
                           DQMOfflinePrePOG *
                           HLTMonitoring *
                           DQMMessageLogger )

DQMOfflineExtraHLT = cms.Sequence( offlineValidationHLTSource )


DQMOfflineFakeHLT = cms.Sequence( DQMOffline )
DQMOfflineFakeHLT.remove( HLTMonitoring )
DQMOfflineFakeHLT.remove( DQMOfflineTrigger )

#MC
DQMOfflinePrePOGMC = cms.Sequence( DQMOfflineVertex *
                                   DQMOfflineBTag *
                                   DQMOfflinePhysics )

DQMOfflinePOGMC = cms.Sequence( DQMOfflinePrePOGMC *
                                DQMMessageLogger )

#DQMOfflineCommon
from DQM.TrackingMonitorSource.pixelTracksMonitoring_cff import *
from DQM.SiOuterTracker.OuterTrackerSourceConfig_cff import *
from Validation.RecoTau.DQMSequences_cfi import *

DQMOfflinePixelTracking = cms.Sequence( pixelTracksMonitoring )

DQMOuterTracker = cms.Sequence( DQMOfflineDCS *
                                OuterTrackerSource *
                                DQMMessageLogger *
                                DQMOfflinePhysics *
                                DQMOfflineVertex 
                                )

DQMOfflineTrackerPhase2 = cms.Sequence( trackerphase2DQMSource )

DQMOfflineTAU = cms.Sequence( produceDenomsData *
				pfTauRunDQMValidation )

DQMOfflineTrackerStripCommon = cms.Sequence( SiStripDQMTier0Common )

DQMOfflineTrackerPixel = cms.Sequence( siPixelOfflineDQM_source )

DQMOfflineCommon = cms.Sequence( DQMOfflineDCS *
                                 DQMMessageLogger *
				 DQMOfflineTrackerStrip * 
				 DQMOfflineTrackerPixel *
                                 DQMOfflineTracking *
                                 DQMOfflineTrigger *
                                 DQMOfflineBeam *
                                 DQMOfflineCASTOR *
                                 DQMOfflinePhysics *
				 DQMOfflineTAU
                                )

DQMOfflineCommonFakeHLT = cms.Sequence( DQMOfflineCommon )
DQMOfflineCommonFakeHLT.remove( DQMOfflineTrigger )

#MinBias/ZeroBias
DQMOfflineTrackerStripMinBias = cms.Sequence( SiStripDQMTier0MinBias )

DQMOfflineTrackingMinBias = cms.Sequence( TrackingDQMSourceTier0MinBias *
                                   DQMOfflineVertex *
                                   materialDumperAnalyzer )


DQMOfflineCommonSiStripZeroBias = cms.Sequence( DQMOfflineDCS *
                                 DQMMessageLogger *
				 DQMOfflineTrackerStripMinBias *
				 DQMOfflineTrackerPixel *
                                 DQMOfflineL1T *
                                 DQMOfflineTrigger *
                                 DQMOfflineBeam *
                                 DQMOfflineCASTOR *
                                 DQMOfflinePhysics *
				 DQMOfflineTrackingMinBias
                                 )

DQMOfflineCommonSiStripZeroBiasFakeHLT = cms.Sequence( DQMOfflineCommonSiStripZeroBias )
DQMOfflineCommonSiStripZeroBiasFakeHLT.remove( DQMOfflineTrigger )

#Other definitons
from DQMOffline.Lumi.ZCounting_cff import *

DQMOfflineLumi = cms.Sequence ( zcounting )

DQMOfflineMuon = cms.Sequence( dtSources *
                               rpcTier0Source *
                               cscSources *
                               muonMonitors
                              )

#Taus not created in pp conditions for HI
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
_DQMOfflineTAU = cms.Sequence()
pp_on_AA_2018.toReplaceWith(DQMOfflineTAU, _DQMOfflineTAU)


# miniAOD DQM validation
from Validation.RecoParticleFlow.miniAODDQM_cff import * # On MiniAOD vs RECO
from Validation.RecoParticleFlow.DQMForPF_MiniAOD_cff import * # MiniAOD PF variables
from DQM.TrackingMonitor.tracksDQMMiniAOD_cff import *
from DQMOffline.Muon.miniAOD_cff import *
from DQM.Physics.DQMTopMiniAOD_cff import *

DQMOfflineMiniAOD = cms.Sequence(jetMETDQMOfflineRedoProductsMiniAOD*muonMonitors_miniAOD*MuonMiniAOD*DQMOfflinePF)

#Post sequences are automatically placed in the EndPath by ConfigBuilder if PAT is run.
#miniAOD DQM sequences need to access the filter results.

PostDQMOfflineMiniAOD = cms.Sequence(miniAODDQMSequence*jetMETDQMOfflineSourceMiniAOD*tracksDQMMiniAOD*topPhysicsminiAOD)
PostDQMOffline = cms.Sequence()

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toReplaceWith( PostDQMOfflineMiniAOD, PostDQMOfflineMiniAOD.copyAndExclude([
    pfMetDQMAnalyzerMiniAOD, pfPuppiMetDQMAnalyzerMiniAOD # No hcalnoise yet
]))

from PhysicsTools.NanoAOD.nanoDQM_cff import nanoDQM
DQMOfflineNanoAOD = cms.Sequence(nanoDQM)
#PostDQMOfflineNanoAOD = cms.Sequence(nanoDQM)
