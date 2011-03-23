import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLogger_cfi import *
from DQMServices.Components.DQMDcsInfo_cfi import *

from DQMOffline.Ecal.ecal_dqm_source_offline_cff import *
from DQM.HcalMonitorModule.hcal_dqm_source_fileT0_HeavyIons_cff import *
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_HeavyIons_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_source_cff import *
from DQM.DTMonitorModule.dtDQMOfflineSources_cff import *
from DQM.RPCMonitorClient.RPCTier0Source_cff import *
from DQM.CSCMonitorModule.csc_dqm_sourceclient_offline_cff import *
from DQM.EcalPreshowerMonitorModule.es_dqm_source_offline_cff import *
from DQM.BeamMonitor.AlcaBeamMonitorHeavyIons_cff import *

DQMOfflineHeavyIonsPreDPG = cms.Sequence( dqmDcsInfo *
                                          ecal_dqm_source_offline *
                                          hcalOfflineDQMSource *
                                          SiStripDQMTier0_hi *
                                          siPixelOfflineDQM_heavyions_source *
                                          dtSources *
                                          rpcTier0Source *
                                          cscSources *
                                          es_dqm_source_offline )

DQMOfflineHeavyIonsDPG = cms.Sequence( DQMOfflineHeavyIonsPreDPG *
                                       DQMMessageLogger )

from DQMOffline.Muon.muonMonitors_cff import *
from DQMOffline.JetMET.jetMETDQMOfflineSourceHI_cff import *
from DQMOffline.EGamma.egammaDQMOffline_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_cff import *
#from DQMOffline.RecoB.PrimaryVertexMonitor_cff import *
from DQM.Physics.DQMPhysics_cff import *

egammaDQMOffline.remove(electronAnalyzerSequence)
photonAnalysis.isHeavyIon = True
photonAnalysis.barrelRecHitProducer = 'ecalRecHit'
photonAnalysis. barrelRecHitCollection = 'EcalRecHitsEB'
photonAnalysis.endcapRecHitProducer = 'ecalRecHit'
photonAnalysis. endcapRecHitCollection = 'EcalRecHitsEE'
hltResults.RecHitsEBTag = cms.untracked.InputTag("ecalRecHit", "EcalRecHitsEB")
hltResults.RecHitsEETag = cms.untracked.InputTag("ecalRecHit", "EcalRecHitsEE")

DQMOfflineHeavyIonsPrePOG = cms.Sequence( muonMonitors 
                                          * jetMETDQMOfflineSource
                                          * egammaDQMOffline
                                          * triggerOfflineDQMSource
                                          #* pvMonitor
                                          * alcaBeamMonitor
                                          * dqmPhysicsHI
                                          )

DQMOfflineHeavyIonsPOG = cms.Sequence( DQMOfflineHeavyIonsPrePOG *
                                       DQMMessageLogger )

DQMOfflineHeavyIons = cms.Sequence( DQMOfflineHeavyIonsPreDPG *
                                    DQMOfflineHeavyIonsPrePOG *
                                    DQMMessageLogger )
    
#DQMOfflineHeavyIonsPhysics = cms.Sequence( dqmPhysics )
