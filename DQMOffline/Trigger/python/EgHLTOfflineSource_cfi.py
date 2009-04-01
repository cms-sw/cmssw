import FWCore.ParameterSet.Config as cms


from DQMOffline.Trigger.EgHLTOffEleSelection_cfi import *
from DQMOffline.Trigger.EgHLTOffPhoSelection_cfi import *
from DQMOffline.Trigger.EgHLTOffTrigSelection_cfi import *
from DQMOffline.Trigger.EgHLTOffHistBins_cfi import *
from DQMOffline.Trigger.EgHLTOffFiltersToMon_cfi import *

egHLTOffDQMSource = cms.EDFilter("EgHLTOfflineSource",
                                 egHLTOffFiltersToMon,
                                 binData = cms.PSet(egHLTOffDQMBinData,),

                                 #products we need
                                 triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
                                 EndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                                 BarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                                 ElectronCollection = cms.InputTag("gsfElectrons"),
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


