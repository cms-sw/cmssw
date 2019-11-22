import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLogger_cfi import *
from DQMServices.Components.DQMDcsInfo_cfi import *
from DQMServices.Components.DQMFastTimerService_cff import *

from DQMOffline.Ecal.ecal_dqm_source_offline_cosmic_cff import *
from DQM.HcalTasks.OfflineSourceSequence_cosmic import *
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_Cosmic_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_source_cff import *
from DQM.DTMonitorModule.dtDQMOfflineSources_Cosmics_cff import *
from DQM.RPCMonitorClient.RPCTier0Source_cff import *
from DQM.CSCMonitorModule.csc_dqm_sourceclient_offline_cff import *
from DQM.EcalPreshowerMonitorModule.es_dqm_source_offline_cosmic_cff import *
from DQM.CastorMonitor.castor_dqm_sourceclient_offline_cff import *

DQMOfflineCosmicsDCS = cms.Sequence( dqmDcsInfo )

DQMOfflineCosmicsEcal = cms.Sequence( ecal_dqm_source_offline *
                                es_dqm_source_offline )

DQMOfflineCosmicsHcal = cms.Sequence( hcalOfflineSourceSequence )

DQMOfflineCosmicsTrackerStrip = cms.Sequence( SiStripDQMTier0 )

DQMOfflineCosmicsTrackerPixel = cms.Sequence( siPixelOfflineDQM_cosmics_source )

DQMOfflineCosmicsMuonDPG = cms.Sequence( dtSourcesCosmics *
                                  rpcTier0Source *
                                  cscSources )

DQMOfflineCosmicsCASTOR = cms.Sequence( castorSources )

DQMOfflineCosmicsPreDPG = cms.Sequence( DQMOfflineCosmicsDCS *
                                        DQMOfflineCosmicsEcal *
                                        DQMOfflineCosmicsHcal *
                                        DQMOfflineCosmicsTrackerStrip *
                                        DQMOfflineCosmicsTrackerPixel * 
					DQMOfflineCosmicsMuonDPG *
                                        DQMOfflineCosmicsCASTOR 
					)

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

DQMOfflineCosmicsPOG = cms.Sequence( DQMOfflineCosmicsPrePOG *
                                     DQMMessageLogger )

DQMOfflineCosmics = cms.Sequence( DQMOfflineCosmicsPreDPG *
                                  DQMOfflineCosmicsPrePOG *
                                  DQMMessageLogger )
