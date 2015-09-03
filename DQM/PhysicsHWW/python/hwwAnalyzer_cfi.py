import FWCore.ParameterSet.Config as cms
from RecoJets.JetProducers.PileupJetID_cfi import PhilV1

hwwAnalyzer = cms.EDAnalyzer(
    "HWWAnalyzer",
    doTest                       = cms.bool(False), #True will just access all the collections, False will run the cutflow
    primaryVertexInputTag        = cms.InputTag("offlinePrimaryVertices"),
    trackInputTag                = cms.InputTag("generalTracks"),
    electronsInputTag            = cms.InputTag("gedGsfElectrons"),
    gsftrksInputTag              = cms.InputTag("electronGsfTracks"),
    pfCandsInputTag              = cms.InputTag("particleFlow"),
    recoConversionInputTag       = cms.InputTag("allConversions"),
    cluster1InputTag             = cms.InputTag("reducedEcalRecHitsEB"),
    cluster2InputTag             = cms.InputTag("reducedEcalRecHitsEE"),
    beamSpotTag                  = cms.InputTag("offlineBeamSpot"),
    muonsInputTag                = cms.InputTag("muons"),                         
    muonShower                   = cms.InputTag("muons", "muonShowerInformation"),
    pfJetsInputTag               = cms.InputTag("ak4PFJets"),
    pfElectronsTag               = cms.InputTag("particleFlow","electrons"),
    gsftracksInputTag            = cms.InputTag("electronGsfTracks"),
    rhoInputTag                  = cms.InputTag("kt6PFJetsDeterministicIso","rho"),
    wwrhoInputTag                = cms.InputTag("kt6PFJets","rho"),
    wwrhovorInputTag             = cms.InputTag("kt6PFJetsForRhoComputationVoronoi","rho"),
    forEGIsoInputTag             = cms.InputTag("kt6PFJetsForEGIsolation","rho"),
    pfmetInputTag                = cms.InputTag("pfMet"),
    trackCountingHighEffBJetTags = cms.InputTag("PFTrackCountingHighEffBJetTags"),
    jetCorrector                 = cms.string("ak4PFL1FastL2L3"),

    puJetIDParams = PhilV1,

    InputEGammaWeights1  = cms.FileInPath("DQM/PhysicsHWW/data/Electrons_BDTG_TrigV0_Cat1.weights.xml"),
    InputEGammaWeights2  = cms.FileInPath("DQM/PhysicsHWW/data/Electrons_BDTG_TrigV0_Cat2.weights.xml"),
    InputEGammaWeights3  = cms.FileInPath("DQM/PhysicsHWW/data/Electrons_BDTG_TrigV0_Cat3.weights.xml"),
    InputEGammaWeights4  = cms.FileInPath("DQM/PhysicsHWW/data/Electrons_BDTG_TrigV0_Cat4.weights.xml"),
    InputEGammaWeights5  = cms.FileInPath("DQM/PhysicsHWW/data/Electrons_BDTG_TrigV0_Cat5.weights.xml"),
    InputEGammaWeights6  = cms.FileInPath("DQM/PhysicsHWW/data/Electrons_BDTG_TrigV0_Cat6.weights.xml"),
    InputMuonIsoWeights1 = cms.FileInPath("DQM/PhysicsHWW/data/MuonIsoMVA_sixie-BarrelPt5To10_V0_BDTG.weights.xml"),
    InputMuonIsoWeights2 = cms.FileInPath("DQM/PhysicsHWW/data/MuonIsoMVA_sixie-EndcapPt5To10_V0_BDTG.weights.xml"),
    InputMuonIsoWeights3 = cms.FileInPath("DQM/PhysicsHWW/data/MuonIsoMVA_sixie-BarrelPt10ToInf_V0_BDTG.weights.xml"),
    InputMuonIsoWeights4 = cms.FileInPath("DQM/PhysicsHWW/data/MuonIsoMVA_sixie-EndcapPt10ToInf_V0_BDTG.weights.xml"),
    InputMuonIsoWeights5 = cms.FileInPath("DQM/PhysicsHWW/data/MuonIsoMVA_sixie-Tracker_V0_BDTG.weights.xml"),
    InputMuonIsoWeights6 = cms.FileInPath("DQM/PhysicsHWW/data/MuonIsoMVA_sixie-Global_V0_BDTG.weights.xml")
)
