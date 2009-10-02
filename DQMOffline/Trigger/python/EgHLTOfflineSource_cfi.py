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
                                 hltTag = cms.string("HLT"),
                                 EndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                                 BarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                                 ElectronCollection = cms.InputTag("gsfElectrons"),
                                 PhotonCollection = cms.InputTag("photons"),
                                 CaloJetCollection = cms.InputTag("sisCone5CaloJets"),
                                 IsolTrackCollection = cms.InputTag("generalTracks"),
                                 HBHERecHitCollection = cms.InputTag("hbhereco"),
                                 HFRecHitCollection = cms.InputTag("hfreco"),
                                 DQMDirName=cms.string("HLT/EgOffline"),
                                 
                                 BeamSpotProducer = cms.InputTag("offlineBeamSpot"),
                                 CaloTowers = cms.InputTag("towerMaker"),

                                 #hcal isolation parameters
                                 #first which ones do we want to calculate...
                                 calHLTHcalIsol = cms.bool(False),
                                 calHLTEmIsol = cms.bool(True),
                                 calHLTEleTrkIsol = cms.bool(True),
                                 calHLTPhoTrkIsol = cms.bool(True),
                                 #actual settings for hlt isolation 
                                 hltEMIsolOuterCone = cms.double(0.4),
                                 hltEMIsolInnerConeEB = cms.double(0.045),
                                 hltEMIsolEtaSliceEB = cms.double(0.02),
                                 hltEMIsolEtMinEB = cms.double(-9999.0),
                                 hltEMIsolEMinEB = cms.double(0.08),
                                 hltEMIsolInnerConeEE = cms.double(0.07),
                                 hltEMIsolEtaSliceEE = cms.double(0.02),
                                 hltEMIsolEtMinEE = cms.double(-9999.),
                                 hltEMIsolEMinEE = cms.double(0.3),
                                 hltPhoTrkIsolPtMin = cms.double(1.5),
                                 hltPhoTrkIsolOuterCone = cms.double(0.3),
                                 hltPhoTrkIsolInnerCone = cms.double(0.06),
                                 hltPhoTrkIsolZSpan = cms.double(999999.),
                                 hltPhoTrkIsolRSpan = cms.double(999999.),
                                 hltPhoTrkIsolCountTrks = cms.bool(False),
                                 hltEleTrkIsolPtMin = cms.double(1.5),
                                 hltEleTrkIsolOuterCone = cms.double(0.2),
                                 hltEleTrkIsolInnerCone = cms.double(0.02),
                                 hltEleTrkIsolZSpan = cms.double(0.1),
                                 hltEleTrkIsolRSpan = cms.double(999999.0),
                                 hltHadIsolOuterCone = cms.double(0.3),
                                 hltHadIsolInnerCone = cms.double(0.0),
                                 hltHadIsolEtMin = cms.double(0.),
                                 hltHadIsolDepth = cms.int32(-1),
                                            

                                 
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
                                   cms.PSet (egHLTOffEleEt10SWCuts),
                                   cms.PSet (egHLTOffEleEt15SWCuts),
                                   cms.PSet (egHLTOffEleEt20SWCuts),
                                   cms.PSet (egHLTOffEleEt15SWEleIdCuts),
                                   cms.PSet (egHLTOffEleEt15SWEleIdLTICuts),
                                   cms.PSet (egHLTOffEleEt15SWLTICuts),
                                   cms.PSet (egHLTOffDoubleEleEt10SWCuts),
                                   cms.PSet (egHLTOffPhoEt10Cuts),
                                   cms.PSet (egHLTOffPhoEt15Cuts),
                                   cms.PSet (egHLTOffPhoEt25Cuts),
                                   cms.PSet (egHLTOffPhoEt30Cuts),
                                   cms.PSet (egHLTOffPhoEt25LEITICuts),
                                   
                                 )
                                 
)


