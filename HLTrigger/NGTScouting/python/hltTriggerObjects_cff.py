import FWCore.ParameterSet.Config as cms

# ---------------------------------------------------------------------------
# Example configuration for T
#
# Drop this fragment into your nanoAOD customisation function and add
# trigObjFlatTable to your output commands / schedule.
# ---------------------------------------------------------------------------

hltTriggerObjP4Table = cms.EDProducer(
    "TrigObjP4FlatTableProducer",

    # TriggerEvent summary from HLT (process name must match your HLT menu).
    triggerEvent=cms.InputTag("hltTriggerSummaryAOD", "", "@currentProcess"),

    # Name of the output branch group in the nanoAOD file.
    tableName=cms.string("TriggerObject"),

    # HLT paths for which:
    #   (1) the last save-tags filter seeds the object set (objects passing it
    #       are included as rows of the table), AND
    #   (2) a per-row bool column named after the path is added.
    # Branch names are the path strings with ':', '/', '-', '.' replaced by '_'.
    pathNames=cms.vstring(
        "HLT_AK4PFPuppiJet520",
        "HLT_PFPuppiHT1070",
        "HLT_PFPuppiMETTypeOne140_PFPuppiMHT140",
        "HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepCSV_2p4",
        "HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4",
        "HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4",
        "HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepFlavour_2p4",
        "HLT_Mu50_FromL1TkMuon",
        "HLT_IsoMu24_FromL1TkMuon",
        "HLT_Mu37_Mu27_FromL1TkMuon",
        "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_FromL1TkMuon",
        "HLT_TriMu_10_5_5_DZ_FromL1TkMuon",
        "HLT_Ele32_WPTight_Unseeded",
        "HLT_Ele26_WP70_Unseeded",
        "HLT_Photon108EB_TightID_TightIso_Unseeded",
        "HLT_Photon187_Unseeded",
        "HLT_DoubleEle25_CaloIdL_PMS2_Unseeded",
        "HLT_Diphoton30_23_IsoCaloId_Unseeded",
        "HLT_Ele32_WPTight_L1Seeded",
        "HLT_Ele115_NonIso_L1Seeded",
        "HLT_Ele26_WP70_L1Seeded",
        "HLT_Photon108EB_TightID_TightIso_L1Seeded",
        "HLT_Photon187_L1Seeded",
        "HLT_DoubleEle25_CaloIdL_PMS2_L1Seeded",
        "HLT_DoubleEle23_12_Iso_L1Seeded",
        "HLT_Diphoton30_23_IsoCaloId_L1Seeded",
        "HLT_DoubleMediumChargedIsoPFTauHPS40_eta2p1",
        "HLT_DoubleMediumDeepTauPFTauHPS35_eta2p1",
        "HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
        "HLT_Ele30_WPTight_L1Seeded_LooseDeepTauPFTauHPS30_eta2p1_CrossL1"
    ),

    # Extra HLT filter labels whose passing objects are also included in the
    # table rows (union with the path seeds above, deduplicated).
    # No additional bool column is created for these - they only extend coverage.
    # Useful for intermediate filters, L1-seeded objects, or cross-trigger seeds.
    extraFilters=cms.vstring(
        # e.g. "hltEGL1SingleEGOrFilter",
        # e.g. "hltEle32WPTightGsfTrackIsoFilter",
    ),
)
