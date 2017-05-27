import FWCore.ParameterSet.Config as cms

higDiphotonHLTOfflineSource = cms.EDAnalyzer(
    "HigDiphotonHLTOfflineSource",
    # Used when fetching triggerSummary and triggerResults
    hltProcessName = cms.string("HLT"),
    # HLT paths passing any one of these regular expressions will be included
    hltPathsToCheck = cms.vstring(
        "HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v",
        "HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v",
        "HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v",
        "HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v",
        "HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v",
    ),
    # Location of plots in DQM
    dirname = cms.untracked.string("HLT/Higgs/Diphoton"), 
    verbose = cms.untracked.bool(False), # default: False
    triggerAccept = cms.untracked.bool(True), # default: True 
    triggerResultsToken = cms.InputTag("TriggerResults","","HLT"),
    pvToken = cms.InputTag("offlinePrimaryVertices"),
    photonsToken = cms.InputTag("gedPhotons"),
    # cuts
    photonMinPt = cms.untracked.double(20.0), #GeV
    
)
