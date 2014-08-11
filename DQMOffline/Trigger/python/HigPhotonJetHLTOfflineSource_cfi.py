import FWCore.ParameterSet.Config as cms

higPhotonJetHLTOfflineSource = cms.EDAnalyzer(
    "HigPhotonJetHLTOfflineSource",
    # Used when fetching triggerSummary and triggerResults
    hltProcessName = cms.string("HLT"),
    # HLT paths passing any one of these regular expressions will be included
    hltPathsToCheck = cms.vstring(
        #"HLT_Photon135_v", # standard test 
        "HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_PFMET40_v",
        "HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_PFMET40_v",
        "HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_PFMET40_v",
        "HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_PFMET40_v",
        "HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_PFMET40_v",
        "HLT_Photon135_PFMET40_v",
        "HLT_Photon150_PFMET40_v",
        "HLT_Photon160_PFMET40_v",
        "HLT_Photon250_NoHE_PFMET40_v",
        "HLT_Photon300_NoHE_PFMET40_v",
        "HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_VBF_v",
        "HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_VBF_v",
        "HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_VBF_v",
        "HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_VBF_v",
        "HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_VBF_v",
        "HLT_Photon135_VBF_v",
        "HLT_Photon150_VBF_v",
        "HLT_Photon160_VBF_v",
        "HLT_Photon250_NoHE_VBF_v",
        "HLT_Photon300_NoHE_VBF_v",
    ),

    ## Location of plots in DQM
    dirname = cms.untracked.string("HLT/xshi"), 

    verbose = cms.untracked.bool(True),
    caloJetsToken = cms.InputTag("ak4CaloJets"),
    pvToken = cms.InputTag("offlinePrimaryVertices"),
    photonsToken = cms.InputTag("gedPhotons"),
    pfMetToken = cms.InputTag("pfMet"),
    
    ## All input tags are specified in this pset for convenience
    # inputTags = cms.PSet(
    #     recoMuon       = cms.InputTag("muons"),
    #     beamSpot       = cms.InputTag("offlineBeamSpot"),
    #     offlinePVs     = cms.InputTag("offlinePrimaryVertices"),
    #     triggerSummary = cms.InputTag("hltTriggerSummaryAOD"),
    #     triggerResults = cms.InputTag("TriggerResults")
    # ),
    ## Both 1D and 2D plots use the binnings defined here
    # binParams = cms.untracked.PSet(
    #     ## parameters for fixed-width plots
    #     NVertex    = cms.untracked.vdouble( 20,  1,   50),
    # ),

    
)
