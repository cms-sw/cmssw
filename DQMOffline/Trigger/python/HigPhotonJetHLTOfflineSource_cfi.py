import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
higPhotonJetHLTOfflineSource = DQMEDAnalyzer(
    "HigPhotonJetHLTOfflineSource",
    # Used when fetching triggerSummary and triggerResults
    hltProcessName = cms.string("HLT"),
    # HLT paths passing any one of these regular expressions will be included
    hltPathsToCheck = cms.vstring(
        "HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_PFMET40_v",
        "HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_VBF_v",
        "HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_PFMET40_v",
        "HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_VBF_v",
        "HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_PFMET40_v",
        "HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_VBF_v",
        "HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_PFMET40_v",
        "HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_VBF_v",
        "HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_PFMET40_v",
        "HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_VBF_v",
        "HLT_Photon120_R9Id90_HE10_Iso40_EBOnly_PFMET40_v",
        "HLT_Photon120_R9Id90_HE10_Iso40_EBOnly_VBF_v",
        "HLT_Photon135_PFMET40_v",
        "HLT_Photon135_VBF_v",
        "HLT_Photon155_v",
        "HLT_Photon250_NoHE_v",
        "HLT_Photon300_NoHE_v",
    ),
    # Location of plots in DQM
#    dirname = cms.untracked.string("HLT/Higgs/PhotonJet"), 
    dirname = cms.untracked.string("HLT/HIG/PhotonJet"), 
    verbose = cms.untracked.bool(False), # default: False
    perLSsaving = cms.untracked.bool(False), #driven by DQMServices/Core/python/DQMStore_cfi.py
    triggerAccept = cms.untracked.bool(True), # default: True 
    triggerResultsToken = cms.InputTag("TriggerResults","","HLT"),
    pvToken = cms.InputTag("offlinePrimaryVertices"),
    photonsToken = cms.InputTag("gedPhotons"),
    pfMetToken = cms.InputTag("pfMet"),
    pfJetsToken = cms.InputTag("ak4PFJets"),

    # cuts
    pfjetMinPt = cms.untracked.double(30.0), #GeV
    photonMinPt = cms.untracked.double(20.0), #GeV
    
)
