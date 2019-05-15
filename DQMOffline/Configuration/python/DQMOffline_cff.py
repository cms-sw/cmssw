import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLogger_cfi import *
from DQMServices.Components.DQMDcsInfo_cfi import *
from DQMServices.Components.DQMFastTimerService_cff import *

from DQMOffline.Ecal.ecal_dqm_source_offline_cff import *
from DQM.HcalTasks.OfflineSourceSequence_pp import *
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_source_cff import *
from DQM.DTMonitorModule.dtDQMOfflineSources_cff import *
from DQM.RPCMonitorClient.RPCTier0Source_cff import *
from DQM.CSCMonitorModule.csc_dqm_sourceclient_offline_cff import *
from DQM.EcalPreshowerMonitorModule.es_dqm_source_offline_cff import *
from DQM.BeamMonitor.AlcaBeamMonitor_cff import *
from DQM.CastorMonitor.castor_dqm_sourceclient_offline_cff import *
from Validation.RecoTau.DQMSequences_cfi import *
from DQMOffline.Hcal.HcalDQMOfflineSequence_cff import *
from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *
from DQM.CTPPS.ctppsDQM_cff import *

DQMNone = cms.Sequence()

DQMOfflinePreDPG = cms.Sequence( dqmDcsInfo *
                                 l1TriggerDqmOffline * # L1 emulator is run within this sequence for real data
                                 ecal_dqm_source_offline *
                                 hcalOfflineSourceSequence *
                                 SiStripDQMTier0 *
                                 siPixelOfflineDQM_source *
                                 dtSources *
                                 rpcTier0Source *
                                 cscSources *
                                 es_dqm_source_offline *
                                 castorSources *
                                 HcalDQMOfflineSequence )

DQMOfflineDPG = cms.Sequence( DQMOfflinePreDPG *
                              DQMMessageLogger )

from DQMOffline.Muon.muonMonitors_cff import *
from DQMOffline.JetMET.jetMETDQMOfflineSource_cff import *
from DQMOffline.EGamma.egammaDQMOffline_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_cff import *
from DQMOffline.RecoB.PrimaryVertexMonitor_cff import *
from DQMOffline.RecoB.dqmAnalyzer_cff import *
from DQMOffline.Lumi.ZCounting_cff import *
from DQM.Physics.DQMPhysics_cff import *
from DQM.Physics.DQMTopMiniAOD_cff import *
from Validation.RecoTau.DQMSequences_cfi import *
from DQM.TrackingMonitorSource.TrackingSourceConfig_Tier0_cff import *
from DQM.TrackingMonitorSource.pixelTracksMonitoring_cff import *
from DQMOffline.RecoB.PixelVertexMonitor_cff import *
from DQM.SiOuterTracker.OuterTrackerSourceConfig_cff import *
# miniAOD DQM validation
from Validation.RecoParticleFlow.miniAODDQM_cff import *
from DQM.TrackingMonitor.tracksDQMMiniAOD_cff import * 
from DQM.TrackingMonitor.trackingRecoMaterialAnalyzer_cfi import materialDumperAnalyzer
materialDumperAnalyzer.usePV = True

DQMOfflinePrePOG = cms.Sequence( TrackingDQMSourceTier0 *
                                 muonMonitors *
                                 jetMETDQMOfflineSource *
                                 egammaDQMOffline *
                                 triggerOfflineDQMSource *
                                 pvMonitor *
                                 materialDumperAnalyzer *
                                 bTagPlotsDATA *
                                 alcaBeamMonitor *
                                 dqmPhysics *
                                 produceDenoms *
                                 pfTauRunDQMValidation)
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel

DQMOfflinePOG = cms.Sequence( DQMOfflinePrePOG *
                              DQMMessageLogger )

HLTMonitoring = cms.Sequence( OfflineHLTMonitoring )
HLTMonitoringPA = cms.Sequence( OfflineHLTMonitoringPA )
DQMOffline = cms.Sequence( DQMOfflinePreDPG *
                           DQMOfflinePrePOG *
                           HLTMonitoring *
                           # dqmFastTimerServiceLuminosity *
                           DQMMessageLogger )

_ctpps_2016_DQMOffline = DQMOffline.copy()
_ctpps_2016_DQMOffline *= ctppsDQM
from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
ctpps_2016.toReplaceWith(DQMOffline, _ctpps_2016_DQMOffline)

_ctpps_2016_DQMOffline = DQMOffline.copy()
#_ctpps_2016_DQMOffline *= ctppsDQM
ctpps_2016.toReplaceWith(DQMOffline, _ctpps_2016_DQMOffline)

DQMOfflineExtraHLT = cms.Sequence(
    offlineValidationHLTSource
)


DQMOfflineFakeHLT = cms.Sequence( DQMOffline )
DQMOfflineFakeHLT.remove( HLTMonitoring )

DQMOfflinePrePOGMC = cms.Sequence( pvMonitor *
                                   bTagPlotsDATA *
                                   dqmPhysics )

DQMOfflinePOGMC = cms.Sequence( DQMOfflinePrePOGMC *
                                DQMMessageLogger )

DQMOfflinePhysics = cms.Sequence( dqmPhysics )



DQMOfflineTracking = cms.Sequence( TrackingDQMSourceTier0Common *
                                   pvMonitor *
                                   materialDumperAnalyzer
                                 )

DQMOfflinePixelTracking = cms.Sequence( pixelTracksMonitoring +
                                        pixelPVMonitor )

DQMOuterTracker = cms.Sequence( dqmDcsInfo *
                                OuterTrackerSource *
                                DQMMessageLogger *
                                dqmPhysics *
                                pvMonitor *
                                produceDenoms
                                )

DQMOfflineCommon = cms.Sequence( dqmDcsInfo *
                                 DQMMessageLogger *
                                 SiStripDQMTier0Common *
                                 siPixelOfflineDQM_source *
                                 DQMOfflineTracking *
                                 triggerOfflineDQMSource *
                                 alcaBeamMonitor *
                                 castorSources *
                                 dqmPhysics *
                                 produceDenoms *
                                 pfTauRunDQMValidation
                                )
DQMOfflineCommonSiStripZeroBias = cms.Sequence( dqmDcsInfo *
                                 DQMMessageLogger *
                                 SiStripDQMTier0MinBias *
                                 TrackingDQMSourceTier0MinBias *
                                 siPixelOfflineDQM_source *
                                 l1TriggerDqmOffline *
                                 triggerOfflineDQMSource *
                                 alcaBeamMonitor *
                                 castorSources *
                                 dqmPhysics *
                                 pvMonitor *
                                 materialDumperAnalyzer *
                                 produceDenoms *
                                 pfTauRunDQMValidation
                                 )
DQMOfflineLumi = cms.Sequence ( zcounting )

muonRecoAnalyzer.doMVA =         cms.bool( True )
muonRecoAnalyzer_miniAOD.doMVA = cms.bool( True )

DQMOfflineMuon = cms.Sequence( dtSources *
                               rpcTier0Source *
                               cscSources *
                               muonMonitors
                              )

DQMOfflineHcal = cms.Sequence( hcalOfflineSourceSequence )

DQMOfflineEcal = cms.Sequence( ecal_dqm_source_offline *
                               es_dqm_source_offline
                             )
DQMOfflineJetMET = cms.Sequence( jetMETDQMOfflineSource )

DQMOfflineEGamma = cms.Sequence( egammaDQMOffline )

DQMOfflineBTag = cms.Sequence( bTagPlotsDATA )

from DQMOffline.Muon.miniAOD_cff import *

DQMOfflineMiniAOD = cms.Sequence(jetMETDQMOfflineRedoProductsMiniAOD*muonMonitors_miniAOD*MuonMiniAOD)

#Post sequences are automatically placed in the EndPath by ConfigBuilder if PAT is run.
#miniAOD DQM sequences need to access the filter results.


PostDQMOfflineMiniAOD = cms.Sequence(miniAODDQMSequence*jetMETDQMOfflineSourceMiniAOD*tracksDQMMiniAOD*topPhysicsminiAOD)
PostDQMOffline = cms.Sequence()

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toReplaceWith( PostDQMOfflineMiniAOD, PostDQMOfflineMiniAOD.copyAndExclude([
    pfMetDQMAnalyzerMiniAOD, pfPuppiMetDQMAnalyzerMiniAOD # No hcalnoise yet
]))

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
_pfTauRunDQMValidation = cms.Sequence()
pp_on_AA_2018.toReplaceWith(pfTauRunDQMValidation, _pfTauRunDQMValidation)

from PhysicsTools.NanoAOD.nanoDQM_cff import nanoDQM
DQMOfflineNanoAOD = cms.Sequence(nanoDQM)
#PostDQMOfflineNanoAOD = cms.Sequence(nanoDQM)

# L1 trigger sequences
DQMOfflineL1TMonitoring = cms.Sequence( l1TriggerDqmOffline ) # L1 emulator is run within this sequence for real data

DQMOfflineL1TEgamma = cms.Sequence( l1TriggerEgDqmOffline )

DQMOfflineL1TMuon = cms.Sequence( l1TriggerMuonDqmOffline )
