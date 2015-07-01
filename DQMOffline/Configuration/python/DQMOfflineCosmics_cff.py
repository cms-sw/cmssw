import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLogger_cfi import *
from DQMServices.Components.DQMDcsInfo_cfi import *
from DQMServices.Components.DQMFastTimerService_cff import *

from DQMOffline.Ecal.ecal_dqm_source_offline_cosmic_cff import *
from DQM.HcalMonitorModule.hcal_dqm_source_fileT0_cff import *
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_Cosmic_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_source_cff import *
from DQM.DTMonitorModule.dtDQMOfflineSources_Cosmics_cff import *
from DQM.RPCMonitorClient.RPCTier0Source_cff import *
from DQM.CSCMonitorModule.csc_dqm_sourceclient_offline_cff import *
from DQM.EcalPreshowerMonitorModule.es_dqm_source_offline_cosmic_cff import *
from DQM.CastorMonitor.castor_dqm_sourceclient_offline_cff import *

DQMOfflineCosmicsPreDPG = cms.Sequence( dqmDcsInfo *
                                        ecal_dqm_source_offline *
                                        hcalOfflineDQMSource *
                                        SiStripDQMTier0 *
                                        siPixelOfflineDQM_cosmics_source *
                                        dtSourcesCosmics *
                                        rpcTier0Source *
                                        cscSources *
                                        es_dqm_source_offline *
                                        castorSources )

DQMOfflineCosmicsDPG = cms.Sequence( DQMOfflineCosmicsPreDPG *
                                     DQMMessageLogger )

from DQMOffline.Muon.muonCosmicMonitors_cff import *
from DQMOffline.JetMET.jetMETDQMOfflineSourceCosmic_cff import *
from DQMOffline.EGamma.cosmicPhotonAnalyzer_cff import *
from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_cosmics_cff import *
from DQM.Physics.DQMPhysics_cff import *

DQMOfflineCosmicsPrePOG = cms.Sequence( muonCosmicMonitors *
                                        jetMETDQMOfflineSourceCosmic *
                                        egammaCosmicPhotonMonitors *
#                                        l1TriggerDqmOffline * 
                                        triggerCosmicOfflineDQMSource *
                                        dqmPhysicsCosmics
                                        )

DQMOfflineCosmicsPOG = cms.Sequence( DQMOfflineCosmicsPrePOG *
                                     DQMMessageLogger )

DQMOfflineCosmics = cms.Sequence( DQMOfflineCosmicsPreDPG *
                                  DQMOfflineCosmicsPrePOG *
                                  DQMMessageLogger )

DQMOfflineCosmicsPhysics = cms.Sequence( dqmPhysicsCosmics )

#DQMOfflineCosmicsNoHWW = cms.Sequence(DQMOfflineCosmics)
#DQMOfflineCosmicsNoHWW.remove(hwwCosmicDQM)

