import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLogger_cfi import *
from DQMServices.Components.DQMProvInfo_cfi import *
from DQMServices.Components.DQMFastTimerService_cff import *

from DQMOffline.Ecal.ecal_dqm_source_offline_cosmic_cff import *
from DQM.HcalTasks.OfflineSourceSequence_cosmic import *
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_Cosmic_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_source_cff import *
from DQM.SiTrackerPhase2.Phase2TrackerDQMFirstStep_cff import *
from DQM.DTMonitorModule.dtDQMOfflineSources_Cosmics_cff import *
from DQM.RPCMonitorClient.RPCTier0Source_cff import *
from DQM.CSCMonitorModule.csc_dqm_sourceclient_offline_cff import *
from DQM.GEM.gem_dqm_offline_source_cosmics_cff import *
from DQM.EcalPreshowerMonitorModule.es_dqm_source_offline_cosmic_cff import *
from DQM.CastorMonitor.castor_dqm_sourceclient_offline_cff import *

DQMNone = cms.Sequence()

dqmProvInfo.runType = "cosmics_run"
dqmProvInfo.dcsRecord = cms.untracked.InputTag("onlineMetaDataDigis")
DQMOfflineCosmicsDCS = cms.Sequence( dqmProvInfo )

DQMOfflineCosmicsEcal = cms.Sequence( ecal_dqm_source_offline *
                                es_dqm_source_offline )

DQMOfflineCosmicsHcal = cms.Sequence( hcalOfflineSourceSequence )

DQMOfflineCosmicsTrackerStrip = cms.Sequence( SiStripDQMTier0 )

DQMOfflineCosmicsTrackerPixel = cms.Sequence( siPixelOfflineDQM_cosmics_source )

DQMOfflineCosmicsTrackerPhase2 = cms.Sequence( trackerphase2DQMSource )

#tnp modules are meant for collisions only (DT has separate cff for cosmics)
if cscSources.contains(cscTnPEfficiencyMonitor):
    cscSources.remove(cscTnPEfficiencyMonitor)

if rpcTier0Source.contains(rpcTnPEfficiencyMonitor):
    rpcTier0Source.remove(rpcTnPEfficiencyMonitor)

DQMOfflineCosmicsMuonDPG = cms.Sequence( dtSourcesCosmics *
                                  rpcTier0Source *
                                  cscSources )

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
_run3_GEM_DQMOfflineCosmicsMuonDPG = DQMOfflineCosmicsMuonDPG.copy()
_run3_GEM_DQMOfflineCosmicsMuonDPG += gemSourcesCosmics
(run3_GEM & ~phase2_common).toReplaceWith(DQMOfflineCosmicsMuonDPG, _run3_GEM_DQMOfflineCosmicsMuonDPG)

DQMOfflineCosmicsCASTOR = cms.Sequence( castorSources )

DQMOfflineCosmicsPreDPG = cms.Sequence( DQMOfflineCosmicsDCS *
                                        DQMOfflineCosmicsEcal *
                                        DQMOfflineCosmicsHcal *
                                        DQMOfflineCosmicsTrackerStrip *
                                        DQMOfflineCosmicsTrackerPixel *
					DQMOfflineCosmicsMuonDPG *
                                        DQMOfflineCosmicsCASTOR
					)

# No Strip detector in Phase-2 Tracker
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(DQMOfflineCosmicsPreDPG,DQMOfflineCosmicsPreDPG.copyAndExclude([DQMOfflineCosmicsTrackerStrip,  DQMOfflineCosmicsTrackerPixel]))
         
_DQMOfflineCosmicsPreDPG = DQMOfflineCosmicsPreDPG.copy()
_DQMOfflineCosmicsPreDPG += DQMOfflineCosmicsTrackerPhase2
phase2_tracker.toReplaceWith(DQMOfflineCosmicsPreDPG,_DQMOfflineCosmicsPreDPG)

DQMOfflineCosmicsDPG = cms.Sequence( DQMOfflineCosmicsPreDPG *
                                     DQMMessageLogger )

from DQM.TrackingMonitorSource.TrackingSourceConfig_Tier0_Cosmic_cff import *
from DQMOffline.Muon.muonCosmicMonitors_cff import *
from DQMOffline.JetMET.jetMETDQMOfflineSourceCosmic_cff import *
from DQMOffline.EGamma.cosmicPhotonAnalyzer_cff import *
from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_cosmics_cff import *
from DQM.Physics.DQMPhysics_cff import *

DQMOfflineCosmicsTracking = cms.Sequence( TrackingDQMTier0 )

DQMOfflineCosmicsMUO = cms.Sequence( muonCosmicMonitors )

DQMOfflineCosmicsJetMET = cms.Sequence( jetMETDQMOfflineSourceCosmic )

DQMOfflineCosmicsEGamma = cms.Sequence( egammaCosmicPhotonMonitors )

DQMOfflineCosmicsTrigger = cms.Sequence( l1TriggerDqmOfflineCosmics *
					 triggerCosmicOfflineDQMSource )

DQMOfflineCosmicsPhysics = cms.Sequence( dqmPhysicsCosmics )

DQMOfflineCosmicsPrePOG = cms.Sequence( DQMOfflineCosmicsTracking *
                                        DQMOfflineCosmicsMUO *
# Following modules removed since they produce empty histograms
# and are not used in DC
#                                        DQMOfflineCosmicsJetMET *
#                                        DQMOfflineCosmicsEGamma *
                                        DQMOfflineCosmicsTrigger
#					DQMOfflineCosmicsPhysics
                                        )

phase2_common.toReplaceWith(DQMOfflineCosmicsPrePOG,DQMOfflineCosmicsPrePOG.copyAndExclude([DQMOfflineCosmicsTrigger]))

DQMOfflineCosmicsPOG = cms.Sequence( DQMOfflineCosmicsPrePOG *
                                     DQMMessageLogger )

DQMOfflineCosmics = cms.Sequence( DQMOfflineCosmicsPreDPG *
                                  DQMOfflineCosmicsPrePOG *
                                  DQMMessageLogger )

HLTMonitoring = cms.Sequence( OfflineHLTMonitoring )

PostDQMOffline = cms.Sequence()
