import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLogger_cfi import *
from DQMServices.Components.DQMDcsInfo_cfi import *
from DQMServices.Components.DQMFastTimerService_cff import *

from DQMOffline.Ecal.ecal_dqm_source_offline_HI_cff import *
from DQM.HcalTasks.OfflineSourceSequence_hi import *
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_HeavyIons_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_source_cff import *
from DQM.DTMonitorModule.dtDQMOfflineSources_HI_cff import *
from DQM.RPCMonitorClient.RPCTier0Source_cff import *
from DQM.CSCMonitorModule.csc_dqm_sourceclient_offline_cff import *
from DQM.EcalPreshowerMonitorModule.es_dqm_source_offline_cff import *
from DQM.BeamMonitor.AlcaBeamMonitorHeavyIons_cff import *
from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *

DQMOfflineHeavyIonsPreDPG = cms.Sequence( dqmDcsInfo *
                                          l1TriggerDqmOffline * # L1 emulator is run within this sequence for real data
                                          ecal_dqm_source_offline *
                                          #hcalOfflineSourceSequence *
                                          SiStripDQMTier0_hi *
                                          siPixelOfflineDQM_heavyions_source *
                                          dtSources *
                                          rpcTier0Source *
                                          cscSources *
                                          es_dqm_source_offline )

DQMOfflineHeavyIonsDPG = cms.Sequence( DQMOfflineHeavyIonsPreDPG *
                                       DQMMessageLogger )

from DQMOffline.Muon.muonMonitors_cff import *
diMuonHistos.etaBin = cms.int32(70) #dimuonhistograms mass, bin   
diMuonHistos.etaBBin = cms.int32(70)    
diMuonHistos.etaEBin = cms.int32(70)    
diMuonHistos.LowMassMin = cms.double(2.0)   
diMuonHistos.LowMassMax = cms.double(14.0)    
diMuonHistos.HighMassMin = cms.double(55.0)   
diMuonHistos.HighMassMax = cms.double(125.0)
from DQMOffline.JetMET.jetMETDQMOfflineSourceHI_cff import *
from DQMOffline.EGamma.egammaDQMOffline_cff import *
from DQMOffline.Trigger.DQMOffline_Trigger_cff import *
from DQMOffline.RecoB.PrimaryVertexMonitor_cff import *
from DQM.Physics.DQMPhysics_cff import *
from DQM.TrackingMonitorSource.TrackingSourceConfig_Tier0_HeavyIons_cff import *


triggerOfflineDQMSource.remove(jetMETHLTOfflineAnalyzer)
triggerOfflineDQMSource.remove(exoticaMonitorHLT)

#egammaDQMOffline.remove(electronAnalyzerSequence)
egammaDQMOffline.remove(zmumugammaAnalysis)
egammaDQMOffline.remove(zmumugammaOldAnalysis)
#egammaDQMOffline.remove(photonAnalysis)
photonAnalysis.phoProducer = cms.InputTag("gedPhotonsTmp")
photonAnalysis.isHeavyIon = True
photonAnalysis.barrelRecHitProducer = cms.InputTag("ecalRecHit", "EcalRecHitsEB")
photonAnalysis.endcapRecHitProducer = cms.InputTag("ecalRecHit", "EcalRecHitsEE")

triggerOfflineDQMSource.remove(ak4PFL1FastL2L3CorrectorChain)
from DQMOffline.Trigger.FSQHLTOfflineSource_cfi import getFSQHI
fsqHLTOfflineSource.todo = getFSQHI()


dqmElectronGeneralAnalysis.ElectronCollection = cms.InputTag("gedGsfElectronsTmp")
dqmElectronGeneralAnalysis.TrackCollection = cms.InputTag("hiGeneralTracks")
dqmElectronGeneralAnalysis.VertexCollection = cms.InputTag("hiSelectedVertex")
dqmElectronAnalysisAllElectrons.ElectronCollection = cms.InputTag("gedGsfElectronsTmp")
dqmElectronAnalysisSelectionEt.ElectronCollection = cms.InputTag("gedGsfElectronsTmp")
dqmElectronAnalysisSelectionEtIso.ElectronCollection = cms.InputTag("gedGsfElectronsTmp")
dqmElectronTagProbeAnalysis.ElectronCollection = cms.InputTag("gedGsfElectronsTmp")


stdPhotonAnalysis.isHeavyIon = True
stdPhotonAnalysis.barrelRecHitProducer = cms.InputTag("ecalRecHit", "EcalRecHitsEB")
stdPhotonAnalysis.endcapRecHitProducer = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
hltResults.RecHitsEBTag = cms.untracked.InputTag("ecalRecHit", "EcalRecHitsEB")
hltResults.RecHitsEETag = cms.untracked.InputTag("ecalRecHit", "EcalRecHitsEE")


globalAnalyzer.inputTags.offlinePVs = cms.InputTag("hiSelectedVertex")
trackerAnalyzer.inputTags.offlinePVs = cms.InputTag("hiSelectedVertex")
tightAnalyzer.inputTags.offlinePVs = cms.InputTag("hiSelectedVertex")
looseAnalyzer.inputTags.offlinePVs = cms.InputTag("hiSelectedVertex")

pvMonitor.vertexLabel = cms.InputTag("hiSelectedVertex")


DQMOfflineHeavyIonsPrePOG = cms.Sequence( muonMonitors
                                          * TrackMonDQMTier0_hi
                                          * jetMETDQMOfflineSource
                                          * egammaDQMOffline
                                          * triggerOfflineDQMSource
                                          * pvMonitor
                                          * alcaBeamMonitor
                                          * dqmPhysicsHI
                                          )

#disabled, until an appropriate configuration is set
hltTauOfflineMonitor_PFTaus.Matching.doMatching = False

DQMOfflineHeavyIonsPOG = cms.Sequence( DQMOfflineHeavyIonsPrePOG *
                                       DQMMessageLogger )

DQMOfflineHeavyIons = cms.Sequence( DQMOfflineHeavyIonsPreDPG *
                                    DQMOfflineHeavyIonsPrePOG *
                                    DQMMessageLogger )

#this is needed to have a light sequence for T0 processing
liteDQMOfflineHeavyIons = cms.Sequence ( DQMOfflineHeavyIons )
liteDQMOfflineHeavyIons.remove( SiStripMonitorCluster )
liteDQMOfflineHeavyIons.remove( jetMETDQMOfflineSource )


#DQMOfflineHeavyIonsPhysics = cms.Sequence( dqmPhysics )
