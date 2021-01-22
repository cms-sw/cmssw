import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMMessageLogger_cfi import *
from DQMServices.Components.DQMProvInfo_cfi import *
from DQMServices.Components.DQMFastTimerService_cff import *

from DQMOffline.L1Trigger.L1TriggerDqmOffline_cff import *
from DQMOffline.Ecal.ecal_dqm_source_offline_HI_cff import *
from DQM.EcalPreshowerMonitorModule.es_dqm_source_offline_cff import *
from DQM.HcalTasks.OfflineSourceSequence_hi import *
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_HeavyIons_cff import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_source_cff import *
from DQM.DTMonitorModule.dtDQMOfflineSources_HI_cff import *
from DQM.RPCMonitorClient.RPCTier0Source_cff import *
from DQM.CSCMonitorModule.csc_dqm_sourceclient_offline_cff import *
from DQMOffline.Muon.gem_dqm_offline_source_cff import *
from DQM.BeamMonitor.AlcaBeamMonitorHeavyIons_cff import *

DQMNone = cms.Sequence()

dqmProvInfo.runType = "hi_run"
DQMOfflineHeavyIonsDCS = cms.Sequence( dqmProvInfo )

# L1 trigger sequences
DQMOfflineHeavyIonsL1T = cms.Sequence( l1TriggerDqmOffline ) # L1 emulator is run within this sequence for real data

#DPGs
DQMOfflineHeavyIonsEcal = cms.Sequence( ecal_dqm_source_offline *
                                es_dqm_source_offline )

DQMOfflineHeavyIonsHcal = cms.Sequence( hcalOfflineSourceSequence )

DQMOfflineHeavyIonsTrackerStrip = cms.Sequence( SiStripDQMTier0_hi )

DQMOfflineHeavyIonsTrackerPixel = cms.Sequence( siPixelOfflineDQM_heavyions_source )

DQMOfflineHeavyIonsMuonDPG = cms.Sequence( dtSources *
                                  rpcTier0Source *
                                  cscSources )

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
_run3_GEM_DQMOfflineHeavyIonsMuonDPG = DQMOfflineHeavyIonsMuonDPG.copy()
_run3_GEM_DQMOfflineHeavyIonsMuonDPG += gemSources
run3_GEM.toReplaceWith(DQMOfflineHeavyIonsMuonDPG, _run3_GEM_DQMOfflineHeavyIonsMuonDPG)

DQMOfflineHeavyIonsPreDPG = cms.Sequence( DQMOfflineHeavyIonsDCS *
					  DQMOfflineHeavyIonsL1T *
					  DQMOfflineHeavyIonsEcal *
					  DQMOfflineHeavyIonsHcal *
					  DQMOfflineHeavyIonsTrackerStrip *
                                          DQMOfflineHeavyIonsTrackerPixel *
					  DQMOfflineHeavyIonsMuonDPG )

DQMOfflineHeavyIonsDPG = cms.Sequence( DQMOfflineHeavyIonsPreDPG *
                                       DQMMessageLogger )

#Modifications
from DQMOffline.Muon.muonMonitors_cff import *
diMuonHistos.etaBin = cms.int32(70) #dimuonhistograms mass, bin   
diMuonHistos.etaBBin = cms.int32(70)    
diMuonHistos.etaEBin = cms.int32(70)    
diMuonHistos.LowMassMin = cms.double(2.0)   
diMuonHistos.LowMassMax = cms.double(14.0)    
diMuonHistos.HighMassMin = cms.double(55.0)   
diMuonHistos.HighMassMax = cms.double(125.0)

from DQMOffline.Trigger.DQMOffline_Trigger_cff import *
triggerOfflineDQMSource.remove(jetMETHLTOfflineAnalyzer)
triggerOfflineDQMSource.remove(exoticaMonitorHLT)
triggerOfflineDQMSource.remove(susyMonitorHLT)
triggerOfflineDQMSource.remove(b2gMonitorHLT)
triggerOfflineDQMSource.remove(bphMonitorHLT)
triggerOfflineDQMSource.remove(higgsMonitorHLT)
triggerOfflineDQMSource.remove(smpMonitorHLT)
triggerOfflineDQMSource.remove(topMonitorHLT)
triggerOfflineDQMSource.remove(btagMonitorHLT)
triggerOfflineDQMSource.remove(egammaMonitorHLT)
triggerOfflineDQMSource.remove(ak4PFL1FastL2L3CorrectorChain)

globalAnalyzerTnP.inputTags.offlinePVs = cms.InputTag("hiSelectedVertex")
trackerAnalyzerTnP.inputTags.offlinePVs = cms.InputTag("hiSelectedVertex")
tightAnalyzerTnP.inputTags.offlinePVs = cms.InputTag("hiSelectedVertex")
looseAnalyzerTnP.inputTags.offlinePVs = cms.InputTag("hiSelectedVertex")

from DQMOffline.EGamma.egammaDQMOffline_cff import *
#egammaDQMOffline.remove(electronAnalyzerSequence)
egammaDQMOffline.remove(zmumugammaAnalysis)
egammaDQMOffline.remove(zmumugammaOldAnalysis)
#egammaDQMOffline.remove(photonAnalysis)

photonAnalysis.phoProducer = cms.InputTag("gedPhotonsTmp")
photonAnalysis.isHeavyIon = True
photonAnalysis.barrelRecHitProducer = cms.InputTag("ecalRecHit", "EcalRecHitsEB")
photonAnalysis.endcapRecHitProducer = cms.InputTag("ecalRecHit", "EcalRecHitsEE")

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

#disabled, until an appropriate configuration is set
hltTauOfflineMonitor_PFTaus.Matching.doMatching = False

from DQMOffline.Trigger.FSQHLTOfflineSource_cfi import getFSQHI
fsqHLTOfflineSource.todo = getFSQHI()

from DQMOffline.RecoB.PrimaryVertexMonitor_cff import *
pvMonitor.vertexLabel = cms.InputTag("hiSelectedVertex")

from DQM.TrackingMonitorSource.TrackingSourceConfig_Tier0_HeavyIons_cff import *
from DQMOffline.JetMET.jetMETDQMOfflineSourceHI_cff import *
from DQM.BeamMonitor.AlcaBeamMonitorHeavyIons_cff import *
from DQM.Physics.DQMPhysics_cff import *

DQMOfflineHeavyIonsMUO = cms.Sequence(muonMonitors)

DQMOfflineHeavyIonsTracking = cms.Sequence( TrackMonDQMTier0_hi )

DQMOfflineHeavyIonsJetMET = cms.Sequence( jetMETDQMOfflineSource )

DQMOfflineHeavyIonsEGamma = cms.Sequence( egammaDQMOffline )

DQMOfflineHeavyIonsTrigger = cms.Sequence( triggerOfflineDQMSource )

DQMOfflineHeavyIonsVertex = cms.Sequence( pvMonitor )

DQMOfflineHeavyIonsBeam = cms.Sequence( alcaBeamMonitor )

DQMOfflineHeavyIonsPhysics = cms.Sequence( dqmPhysicsHI )

DQMOfflineHeavyIonsPrePOG = cms.Sequence( DQMOfflineHeavyIonsMUO *
                                           DQMOfflineHeavyIonsTracking *
                                           DQMOfflineHeavyIonsJetMET *
                                           DQMOfflineHeavyIonsEGamma * 
                                           DQMOfflineHeavyIonsTrigger *
                                           DQMOfflineHeavyIonsVertex *
                                           DQMOfflineHeavyIonsBeam *
                                           DQMOfflineHeavyIonsPhysics )

DQMOfflineHeavyIonsPOG = cms.Sequence( DQMOfflineHeavyIonsPrePOG *
                                       DQMMessageLogger )

DQMOfflineHeavyIons = cms.Sequence( DQMOfflineHeavyIonsPreDPG *
                                    DQMOfflineHeavyIonsPrePOG *
                                    DQMMessageLogger )

DQMOfflineHeavyIonsFakeHLT = cms.Sequence( DQMOfflineHeavyIons )
DQMOfflineHeavyIonsFakeHLT.remove( triggerOfflineDQMSource )

#this is needed to have a light sequence for T0 processing
liteDQMOfflineHeavyIons = cms.Sequence ( DQMOfflineHeavyIons )
liteDQMOfflineHeavyIons.remove( SiStripMonitorCluster )
liteDQMOfflineHeavyIons.remove( jetMETDQMOfflineSource )

PostDQMOfflineHI = cms.Sequence()
