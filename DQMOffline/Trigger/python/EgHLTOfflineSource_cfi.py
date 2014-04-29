import FWCore.ParameterSet.Config as cms


from DQMOffline.Trigger.EgHLTOffEleSelection_cfi import *
from DQMOffline.Trigger.EgHLTOffPhoSelection_cfi import *
from DQMOffline.Trigger.EgHLTOffTrigSelection_cfi import *
from DQMOffline.Trigger.EgHLTOffHistBins_cfi import *
from DQMOffline.Trigger.EgHLTOffFiltersToMon_cfi import *

egHLTOffDQMSource = cms.EDAnalyzer("EgHLTOfflineSource",
                                 egHLTOffFiltersToMon,
                                 binData = cms.PSet(egHLTOffDQMBinData,),

                                 #products we need
                                 triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
                                 hltTag = cms.string("HLT"),
                                 TrigResults = cms.InputTag("TriggerResults","","HLT"),
                                 filterInactiveTriggers = cms.bool(True),
                                 EndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                                 BarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                                 ElectronCollection = cms.InputTag("gedGsfElectrons"),
                                 PhotonCollection = cms.InputTag("gedPhotons"),
                                 VertexCollection = cms.InputTag("offlinePrimaryVertices"),
                                 #CaloJetCollection = cms.InputTag("sisCone5CaloJets"),
                                 #--------Morse
                                 CaloJetCollection = cms.InputTag("ak4CaloJets"),
                                 #--------
                                 IsolTrackCollection = cms.InputTag("generalTracks"),
                                 HBHERecHitCollection = cms.InputTag("hbhereco"),
                                 HFRecHitCollection = cms.InputTag("hfreco"),
                                 DQMDirName=cms.string("HLT/EgOffline"),

                                 BeamSpotProducer = cms.InputTag("offlineBeamSpot"),
                                 CaloTowers = cms.InputTag("towerMaker"),

                                 #hcal isolation parameters
                                 #first which ones do we want to calculate...
                                 calHLTHcalIsol = cms.bool(True),
                                 calHLTEmIsol = cms.bool(True),
                                 calHLTEleTrkIsol = cms.bool(True),
                                 calHLTPhoTrkIsol = cms.bool(False),
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
                                    stdEle = cms.string("et:detEta:dEtaIn:dPhiIn:hadem:sigmaIEtaIEta:hltIsolTrksEle:hltIsolHad:hltIsolEm"),
                                    tagEle = cms.string("et:detEta:dEtaIn:dPhiIn:hadem:sigmaIEtaIEta:hltIsolTrksEle:hltIsolHad:hltIsolEm"),
                                    probeEle = cms.string("et:detEta"),
                                    fakeEle = cms.string("et:detEta:hadem"),
                                    trigTPEle = cms.string("detEta:dEtaIn:dPhiIn:hadem:sigmaIEtaIEta:hltIsolTrksEle:hltIsolHad:hltIsolEm"),
                                    trigTPPho = cms.string("hadem:sigmaIEtaIEta:isolEm:isolHad:isolPtTrks"),
                                    stdPho = cms.string("et:detEta:dEtaIn:dPhiIn:hadem:isolEm:isolHad:isolPtTrks"),
                                 ),
                                 eleCuts = cms.PSet (egHLTOffEleCuts,),
                                 eleLooseCuts = cms.PSet(egHLTOffEleLooseCuts,),
                                 phoCuts = cms.PSet(egHLTOffPhoCuts,),
                                 phoLooseCuts = cms.PSet(egHLTOffPhoLooseCuts,),
                                 triggerCuts = cms.VPSet (

                                   #cms.PSet (egHLTOffEleEt10LWCuts), #8E29
                                   #cms.PSet (egHLTOffEleEt15LWCuts),
                                   #cms.PSet (egHLTOffEleEt10LWEleIdCuts),
                                   #cms.PSet (egHLTOffDoubleEleEt5Cuts),
                                   #cms.PSet (egHLTOffEleEt10SWCuts), #1E31
                                   #cms.PSet (egHLTOffEleEt15SWCuts),
                                   #cms.PSet (egHLTOffEleEt20SWCuts),
                                   #cms.PSet (egHLTOffEleEt15SWEleIdCuts),
                                   #cms.PSet (egHLTOffEleEt15SWEleIdLTICuts),
                                   #cms.PSet (egHLTOffEleEt15SWLTICuts),
                                   #cms.PSet (egHLTOffDoubleEleEt10SWCuts),
                                   #cms.PSet (egHLTOffPhoEt10Cuts),
                                   #cms.PSet (egHLTOffPhoEt15Cuts),
                                   #cms.PSet (egHLTOffPhoEt15LEICuts),
                                   #cms.PSet (egHLTOffPhoEt15HTICuts),
                                   #cms.PSet (egHLTOffPhoEt20Cuts),
                                   #cms.PSet (egHLTOffPhoEt25Cuts),
                                   #cms.PSet (egHLTOffPhoEt30Cuts),
                                   #cms.PSet (egHLTOffPhoEt10LEITICuts),
                                   #cms.PSet (egHLTOffPhoEt20LEITICuts),
                                   #cms.PSet (egHLTOffPhoEt25LEITICuts),
                                   #cms.PSet (egHLTOffDoublePhoEt10Cuts),
                                   #cms.PSet (egHLTOffDoublePhoEt15Cuts),
                                   #cms.PSet (egHLTOffDoublePhoEt15VLEICuts),
                                   #----Morse-----------
				   #5E32
                                   #cms.PSet (egHLTOffPhotonEt30_CaloIdVL_v1Cuts),
                                   #cms.PSet (egHLTOffPhotonEt30_CaloIdVL_IsoL_v1Cuts),
                                   #cms.PSet (egHLTOffPhotonEt50_CaloIdVL_IsoL_v1Cuts),
                                   #cms.PSet (egHLTOffPhotonEt75_CaloIdVL_v1Cuts),
                                   #cms.PSet (egHLTOffPhotonEt75_CaloIdVL_IsoL_v1Cuts),
                                   #cms.PSet (egHLTOffPhotonEt125_NoSpikeFilter_v1Cuts),
                                   #cms.PSet (egHLTOffDoublePhotonEt33_v1Cuts),
                                   #cms.PSet (egHLTOffEleEt8_v1Cuts),
                                   #cms.PSet (egHLTOffEleEt8_CaloIdL_CaloIsoVL_v1Cuts),
                                   #cms.PSet (egHLTOffEleEt8_CaloIdL_TrkIdVL_v1Cuts),
                                   #cms.PSet (egHLTOffEleEt15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v1Cuts),
                                   #cms.PSet (egHLTOffEleEt17_CaloIdL_CaloIsoVL_v1Cuts),
                                   #cms.PSet (egHLTOffEleEt27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v1Cuts),
                                   #cms.PSet (egHLTOffEleEt45_CaloIdVT_TrkIdT_v1Cuts),
                                   #cms.PSet (egHLTOffEle90_NoSpikeFilter_v1Cuts)#,
                                   #cms.PSet (egHLTOffPhotonEt32_CaloIdL_PhotonEt26_CaloIdL_v1Cuts),
                                   #cms.PSet (egHLTOffEleEt17_CaloIdL_CaloIsoVL_EleEt8_CaloIdL_CaloIsoVL_v1Cuts)
                                   #-------------

                                 )

)


