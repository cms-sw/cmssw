import FWCore.ParameterSet.Config as cms


from DQMOffline.Trigger.EgHLTOffEleSelection_cfi import *
from DQMOffline.Trigger.EgHLTOffPhoSelection_cfi import *
from DQMOffline.Trigger.EgHLTOffTrigSelection_cfi import *
from DQMOffline.Trigger.EgHLTOffHistBins_cfi import *

egHLTOffDQMSource = cms.EDFilter("EgHLTOfflineSource",
                                
                                 filters = cms.VPSet(),
                                 binData = cms.PSet(egHLTOffDQMBinData,),
                                 triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
                                 EndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                                 BarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                                 ElectronCollection = cms.InputTag("pixelMatchGsfElectrons"),
                                 PhotonCollection = cms.InputTag("photons"),
                                 #CaloJetCollection = cms.InputTag("iterativeCone5CaloJets"),
                                 CaloJetCollection = cms.InputTag("L2L3CorJetIC5Calo"),
                                 IsolTrackCollection = cms.InputTag("generalTracks"),
                                 HBHERecHitCollection = cms.InputTag("hbhereco"),
                                 HFRecHitCollection = cms.InputTag("hfreco"),
                                 DQMDirName=cms.string("HLT/EgammaHLTOffline_egammaHLTDQM"),
                                 hltTag = cms.string("HLT"),
                                 eleEcalIsolTag=cms.InputTag("electronEcalRecHitIsolationLcone"),
                                 eleTrkIsolTag=cms.InputTag("electronTrackIsolationLcone"),
                                 eleHcalDepth1IsolTag=cms.InputTag("electronHcalDepth1TowerIsolationLcone"),
                                 eleHcalDepth2IsolTag=cms.InputTag("electronHcalDepth2TowerIsolationLcone"),
                                 phoIDTag = cms.InputTag("PhotonIDProd","PhotonAssociatedID"),
                              
                                 eleHLTFilterNames=cms.vstring("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter",
                                                               "hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter",
                                                               "hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter",
                                                               "hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter"),
                                 
                                 phoHLTFilterNames=cms.vstring("hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter",
                                                               "hlt1jet30"),
                                 eleTightLooseTrigNames=cms.vstring('hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter:hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter',
                                                                    'hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter',
                                                                    'hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter:hlt1jet30'),
                                 phoTightLooseTrigNames=cms.vstring('hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter:hlt1jet30'),
                                 diEleTightLooseTrigNames=cms.vstring('hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter:hltPreMinBiasEcal'),
                                 diPhoTightLooseTrigNames=cms.vstring(),

                                 
                                 #setting up selection
                                 cutMasks = cms.PSet(
                                    stdEle = cms.string("et:detEta:dEtaIn:dPhiIn:hadem:sigmaIEtaIEta:isolEm:isolHad:isolPtTrks"),
                                    tagEle = cms.string("et:detEta:dEtaIn:dPhiIn:hadem:sigmaIEtaIEta:isolEm:isolHad:isolPtTrks"),
                                    probeEle = cms.string("et:detEta"),
                                    fakeEle = cms.string("et:detEta:hadem"),
                                    stdPho = cms.string("et:detEta:dEtaIn:dPhiIn:hadem:r9:isolEm:isolHad:isolPtTrks"),
                                 ),
                                 eleCuts = cms.PSet (egHLTOffEleCuts,),    
                                 eleLooseCuts = cms.PSet(egHLTOffEleLooseCuts,),
                                 phoCuts = cms.PSet(egHLTOffPhoCuts,),
                                 phoLooseCuts = cms.PSet(egHLTOffPhoLooseCuts,),          
                                 triggerCuts = cms.VPSet (
                                   cms.PSet (egHLTOffEleEt15Cuts),
                                   cms.PSet (egHLTOffEleLWEt15Cuts),
                                   cms.PSet (egHLTOffDoubleEleEt5Cuts)
                                 )
                                 
)


