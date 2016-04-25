import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

from DQMServices.Components.DQMMessageLogger_cfi import *
from DQMServices.Components.DQMDcsInfo_cfi import *
from DQMServices.Components.DQMFastTimerService_cff import *
from DQMServices.Components.DQMFastTimerServiceLuminosity_cfi import *

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
from DQM.Physics.DQMPhysics_cff import *
from DQM.Physics.DQMTopMiniAOD_cff import *
from Validation.RecoTau.DQMSequences_cfi import *
from DQM.TrackingMonitorSource.TrackingSourceConfig_Tier0_cff import *
# miniAOD DQM validation
from Validation.RecoParticleFlow.miniAODDQM_cff import *
from DQM.TrackingMonitor.tracksDQMMiniAOD_cff import *

DQMOfflinePrePOG = cms.Sequence( TrackingDQMSourceTier0 *
                                 muonMonitors *
                                 jetMETDQMOfflineSource *
                                 egammaDQMOffline *
                                 triggerOfflineDQMSource *
                                 pvMonitor *
                                 bTagPlotsDATA *
                                 alcaBeamMonitor *
                                 dqmPhysics *
                                 produceDenoms *
                                 pfTauRunDQMValidation)
eras.phase1Pixel.toReplaceWith(DQMOfflinePrePOG, DQMOfflinePrePOG.copyAndExclude([ # FIXME
    TrackingDQMSourceTier0,  # Tracking DQM needs to be migrated for phase1
    triggerOfflineDQMSource, # No HLT yet for 2017, so no need to run the DQM (avoiding excessive printouts)
    pfTauRunDQMValidation,   # Excessive printouts because 2017 doesn't have HLT yet
]))

DQMOfflinePOG = cms.Sequence( DQMOfflinePrePOG *
                              DQMMessageLogger )

HLTMonitoring = cms.Sequence( OfflineHLTMonitoring )

DQMOffline = cms.Sequence( DQMOfflinePreDPG *
                           DQMOfflinePrePOG *
                           HLTMonitoring *
                           dqmFastTimerServiceLuminosity *
                           DQMMessageLogger )
eras.phase1Pixel.toReplaceWith(DQMOffline, DQMOffline.copyAndExclude([
    HLTMonitoring # No HLT yet for 2017, so no need to run the DQM (avoiding excessive printouts)
]))

DQMOfflineFakeHLT = cms.Sequence( DQMOffline )
DQMOfflineFakeHLT.remove( HLTMonitoring )

DQMOfflinePrePOGMC = cms.Sequence( pvMonitor *
                                   bTagPlotsDATA *
                                   dqmPhysics )

DQMOfflinePOGMC = cms.Sequence( DQMOfflinePrePOGMC *
                                DQMMessageLogger )
    
DQMOfflinePhysics = cms.Sequence( dqmPhysics )


DQMOfflineCommon = cms.Sequence( dqmDcsInfo *
                                 DQMMessageLogger *
                                 SiStripDQMTier0Common *
                                 TrackingDQMSourceTier0Common *
                                 siPixelOfflineDQM_source *
                                 l1TriggerDqmOffline *
                                 triggerOfflineDQMSource *
                                 alcaBeamMonitor *
                                 castorSources *
                                 dqmPhysics *
                                 pvMonitor *
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
                                 produceDenoms *
                                 pfTauRunDQMValidation 
                                 )
DQMOfflineTracking = cms.Sequence( TrackingDQMSourceTier0Common *
                                   pvMonitor
                                 )
eras.phase1Pixel.toReplaceWith(DQMOfflineTracking, DQMOfflineTracking.copyAndExclude([ # FIXME
    TrackingDQMSourceTier0Common # Tracking DQM needs to be migrated for phase1
]))
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
                                                                 
DQMOfflineMiniAOD = cms.Sequence(jetMETDQMOfflineRedoProductsMiniAOD)

#Post sequences are automatically placed in the EndPath by ConfigBuilder if PAT is run.
#miniAOD DQM sequences need to access the filter results.
PostDQMOfflineMiniAOD = cms.Sequence(miniAODDQMSequence*jetMETDQMOfflineSourceMiniAOD*tracksDQMMiniAOD*topPhysicsminiAOD)
PostDQMOffline = cms.Sequence()

