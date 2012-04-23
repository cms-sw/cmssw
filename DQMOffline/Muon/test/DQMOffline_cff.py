#### Modifications needed to run only Offline Muon DQM (by A. Calderon) ####


import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLogger_cfi import *
from DQMServices.Components.DQMDcsInfo_cfi import *

from DQMOffline.Ecal.ecal_dqm_source_offline_cff import *
from DQM.HcalMonitorModule.hcal_dqm_source_fileT0_cff import *
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_source_cff import *
from DQM.DTMonitorModule.dtDQMOfflineSources_cff import *
from DQM.RPCMonitorClient.RPCTier0Source_cff import *
from DQM.CSCMonitorModule.csc_dqm_sourceclient_offline_cff import *
from DQM.EcalPreshowerMonitorModule.es_dqm_source_offline_cff import *
from DQM.BeamMonitor.AlcaBeamMonitor_cff import *
from DQM.CastorMonitor.castor_dqm_sourceclient_offline_cff import *

DQMOfflinePreDPG = cms.Sequence( dqmDcsInfo *
                                 #ecal_dqm_source_offline *
                                 #hcalOfflineDQMSource *
                                 SiStripDQMTier0 *
                                 siPixelOfflineDQM_source )
                                 #dtSources *
                                 #rpcTier0Source *
                                 #cscSources *
                                 #es_dqm_source_offline *
                                 #castorSources )

DQMOfflineDPG = cms.Sequence( DQMOfflinePreDPG *
                              DQMMessageLogger )

from DQMOffline.Muon.muonMonitors_cff import *
from DQMOffline.JetMET.jetMETDQMOfflineSource_cff import *
from DQMOffline.EGamma.egammaDQMOffline_cff import *
from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_cff import *
from DQMOffline.RecoB.PrimaryVertexMonitor_cff import *
from DQMOffline.RecoB.dqmAnalyzer_cff import *
from DQM.Physics.DQMPhysics_cff import *

#DQMOfflinePrePOG = cms.Sequence( muonMonitors *
#                                 jetMETDQMOfflineSource *
#                                 egammaDQMOffline *
#                                 l1TriggerDqmOffline *
#                                 triggerOfflineDQMSource *
#                                 pvMonitor *
#                                 bTagPlots *
#                                 alcaBeamMonitor *
#                                 dqmPhysics )


DQMOfflinePrePOG = cms.Sequence( muonMonitors )


DQMOfflinePOG = cms.Sequence( DQMOfflinePrePOG *
                              DQMMessageLogger )

DQMOffline = cms.Sequence( DQMOfflinePreDPG *
                           DQMOfflinePrePOG *
                           DQMMessageLogger )

DQMOfflinePrePOGMC = cms.Sequence( pvMonitor *
                                   bTagPlots *
                                   dqmPhysics )

DQMOfflinePOGMC = cms.Sequence( DQMOfflinePrePOGMC *
                                DQMMessageLogger )
    
DQMOfflinePhysics = cms.Sequence( dqmPhysics )


DQMOfflineCommon = cms.Sequence( dqmDcsInfo *
                                 DQMMessageLogger *
                                 SiStripDQMTier0 *
                                 siPixelOfflineDQM_source 
                                 #l1TriggerDqmOffline *
                                 #triggerOfflineDQMSource *
                                 #alcaBeamMonitor *
                                 #castorSources *
                                 #piZeroAnalysis *
                                 #dqmPhysics
                                )
#DQMOfflineMuon = cms.Sequence( dtSources *
#                               rpcTier0Source *
#                               cscSources *
#                               muonMonitors
#                              )



DQMOfflineMuon = cms.Sequence ( muonMonitors )


DQMOfflineHcal = cms.Sequence( hcalOfflineDQMSource )

DQMOfflineEcal = cms.Sequence( ecal_dqm_source_offline *
                               es_dqm_source_offline
                             )
DQMOfflineJetMET = cms.Sequence( jetMETDQMOfflineSource )

DQMStepOne_Common = cms.Sequence( DQMOfflineCommon )

DQMStepOne_Common_Muon = cms.Sequence( DQMOfflineCommon *
                                       DQMOfflineMuon
                                     )

DQMStepOne_Common_Hcal_JetMET = cms.Sequence(DQMOfflineCommon*DQMOfflineHcal*DQMOfflineJetMET)

DQMStepOne_Common_Ecal = cms.Sequence(DQMOfflineCommon*DQMOfflineEcal)

DQMStepOne_Common_Ecal_Hcal = cms.Sequence(DQMOfflineCommon*DQMOfflineEcal*DQMOfflineHcal)
                                   
DQMStepOne_Common_Muon_JetMET = cms.Sequence(DQMOfflineCommon*DQMOfflineMuon*DQMOfflineJetMET)
