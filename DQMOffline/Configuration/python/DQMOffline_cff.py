import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLogger_cfi import *
from DQMServices.Components.DQMProvInfo_cfi import *
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
from DQM.GEM.gem_dqm_offline_source_cff import *
from DQM.CastorMonitor.castor_dqm_sourceclient_offline_cff import *
from DQM.CTPPS.ctppsDQM_cff import *
from DQM.SiTrackerPhase2.Phase2TrackerDQMFirstStep_cff import *
from DQM.SiPixelHeterogeneous.SiPixelHeterogenousDQM_FirstStep_cff import *

DQMNone = cms.Sequence()

DQMMessageLoggerSeq = cms.Sequence( DQMMessageLogger )

dqmProvInfo.runType = "pp_run"
dqmProvInfo.dcsRecord = cms.untracked.InputTag("onlineMetaDataDigis")
DQMOfflineDCS = cms.Sequence( dqmProvInfo )

# L1 trigger sequences
DQMOfflineL1T = cms.Sequence( l1TriggerDqmOffline ) # L1 emulator is run within this sequence for real data

DQMOfflineL1TEgamma = cms.Sequence( l1TriggerEgDqmOffline )

DQMOfflineL1TMuon = cms.Sequence( l1TriggerMuonDqmOffline )

DQMOfflineL1TPhase2 = cms.Sequence( Phase2l1TriggerDqmOffline )

#DPGs
DQMOfflineEcalOnly = cms.Sequence(
    ecalOnly_dqm_source_offline +
    es_dqm_source_offline )

DQMOfflineEcal = cms.Sequence(
    ecal_dqm_source_offline +
    es_dqm_source_offline )

from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
phase2_ecal_devel.toReplaceWith(DQMOfflineEcalOnly, DQMOfflineEcalOnly.copyAndExclude([es_dqm_source_offline]))

#offline version of the online DQM: used in validation/certification
DQMOfflineHcal = cms.Sequence( hcalOfflineSourceSequence )

# offline DQM: used in Release validation
DQMOfflineHcal2 = cms.Sequence( HcalDQMOfflineSequence )

DQMOfflineHcalOnly = cms.Sequence( hcalOnlyOfflineSourceSequence )

DQMOfflineHcal2Only = cms.Sequence( RecHitsDQMOffline )

DQMOfflineTrackerStrip = cms.Sequence( SiStripDQMTier0 )

DQMOfflineTrackerPixel = cms.Sequence( 	siPixelOfflineDQM_source )

DQMOfflineMuonDPG = cms.Sequence( dtSources *
                                  rpcTier0Source *
                                  cscSources )

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
_run3_GEM_DQMOfflineMuonDPG = DQMOfflineMuonDPG.copy()
_run3_GEM_DQMOfflineMuonDPG += gemSources
run3_GEM.toReplaceWith(DQMOfflineMuonDPG, _run3_GEM_DQMOfflineMuonDPG)

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
from DQM.Physics.heavyFlavorDQMFirstStep_cff import *

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

DQMOfflineHeavyFlavor = cms.Sequence( heavyFlavorDQMSource )

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
from DQMOffline.RecoB.PixelVertexMonitor_cff import *
from DQM.SiOuterTracker.OuterTrackerSourceConfig_cff import *
from Validation.RecoTau.DQMSequences_cfi import *

DQMOfflinePixelTracking = cms.Sequence( pixelTracksMonitoring *
                                        pixelPVMonitor *
                                        monitorpixelSoASource )

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

_run3_GEM_DQMOfflineMuon = DQMOfflineMuon.copy()
_run3_GEM_DQMOfflineMuon += gemSources
run3_GEM.toReplaceWith(DQMOfflineMuon, _run3_GEM_DQMOfflineMuon)

#Taus not created in pp conditions for HI
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
_DQMOfflineTAU = cms.Sequence()
pp_on_AA.toReplaceWith(DQMOfflineTAU, _DQMOfflineTAU)


# miniAOD DQM validation
from Validation.RecoParticleFlow.miniAODDQM_cff import * # On MiniAOD vs RECO
from Validation.RecoParticleFlow.DQMForPF_MiniAOD_cff import * # MiniAOD PF variables
from DQM.TrackingMonitor.tracksDQMMiniAOD_cff import *
from DQMOffline.RecoB.bTagMiniDQM_cff import *
from DQMOffline.Muon.miniAOD_cff import *
from DQM.Physics.DQMTopMiniAOD_cff import *

DQMOfflineMiniAOD = cms.Sequence(jetMETDQMOfflineRedoProductsMiniAOD*bTagMiniDQMSource*muonMonitors_miniAOD*MuonMiniAOD*DQMOfflinePF)

#Post sequences are automatically placed in the EndPath by ConfigBuilder if PAT is run.
#miniAOD DQM sequences need to access the filter results.

PostDQMOfflineMiniAOD = cms.Sequence(miniAODDQMSequence*jetMETDQMOfflineSourceMiniAOD*tracksDQMMiniAOD*topPhysicsminiAOD)
PostDQMOffline = cms.Sequence()

from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toReplaceWith( PostDQMOfflineMiniAOD, PostDQMOfflineMiniAOD.copyAndExclude([
    pfMetDQMAnalyzerMiniAOD, pfPuppiMetDQMAnalyzerMiniAOD # No hcalnoise (yet)
]))

from PhysicsTools.NanoAOD.nanoDQM_cff import nanoDQM
DQMOfflineNanoAOD = cms.Sequence(nanoDQM)
#PostDQMOfflineNanoAOD = cms.Sequence(nanoDQM)
from PhysicsTools.NanoAOD.nanogenDQM_cff import nanogenDQM
DQMOfflineNanoGen = cms.Sequence(nanogenDQM)
from PhysicsTools.NanoAOD.nanojmeDQM_cff import nanojmeDQM
DQMOfflineNanoJME = cms.Sequence(nanojmeDQM)
