import FWCore.ParameterSet.Config as cms

# anti-e phase-2 tauID mva names
mvaNames_phase2 = dict(
    mvaName_NoEleMatch_woGwoGSF_BL = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_woGwoGSF_BL"),
    mvaName_NoEleMatch_wGwoGSF_BL = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_wGwoGSF_BL"),
    mvaName_woGwGSF_BL = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_woGwGSF_BL"),
    mvaName_wGwGSF_BL = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_wGwGSF_BL"),
    mvaName_NoEleMatch_woGwoGSF_EC = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_woGwoGSF_FWEC"),
    mvaName_NoEleMatch_wGwoGSF_EC = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_wGwoGSF_FWEC"),
    mvaName_woGwGSF_EC = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_woGwGSF_FWEC"),
    mvaName_wGwGSF_EC = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_wGwGSF_FWEC"),
    mvaName_NoEleMatch_woGwoGSF_VFEC = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_woGwoGSF_VFWEC"),
    mvaName_NoEleMatch_wGwoGSF_VFEC = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_wGwoGSF_VFWEC"),
    mvaName_woGwGSF_VFEC = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_woGwGSF_VFWEC"),
    mvaName_wGwGSF_VFEC = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_wGwGSF_VFWEC")
)
# anti-e phase-2 tauID (WPs)
mapping_phase2 = cms.VPSet(
    cms.PSet(
        category = cms.uint32(0), # minMVANoEleMatchWOgWOgsfBL
        cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_woGwoGSF_BL"),
        variable = cms.string("pt")
    ),
    cms.PSet(
        category = cms.uint32(2), # minMVANoEleMatchWgWOgsfBL
        cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_wGwoGSF_BL"),
        variable = cms.string("pt")
    ),
    cms.PSet(
        category = cms.uint32(5), # minMVAWOgWgsfBL
        cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_woGwGSF_BL"),
        variable = cms.string("pt")
    ),
    cms.PSet(
        category = cms.uint32(7), # minMVAWgWgsfBL
        cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_wGwGSF_BL"),
        variable = cms.string("pt")
    ),
    cms.PSet(
        category = cms.uint32(8), # minMVANoEleMatchWOgWOgsfEC
        cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_woGwoGSF_FWEC"),
        variable = cms.string("pt")
    ),
    cms.PSet(
        category = cms.uint32(9), # minMVANoEleMatchWOgWOgsfVFEC
        cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_woGwoGSF_VFWEC"),
        variable = cms.string("pt")
    ),
    cms.PSet(
        category = cms.uint32(10), # minMVANoEleMatchWgWOgsfEC
        cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_wGwoGSF_FWEC"),
        variable = cms.string("pt")
    ),
    cms.PSet(
        category = cms.uint32(11), # minMVANoEleMatchWgWOgsfVFEC
        cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_wGwoGSF_VFWEC"),
        variable = cms.string("pt")
    ),
    cms.PSet(
        category = cms.uint32(13), # minMVAWOgWgsfEC
        cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_woGwGSF_FWEC"),
        variable = cms.string("pt")
    ),
    cms.PSet(
        category = cms.uint32(14), # minMVAWOgWgsfVFEC
        cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_woGwGSF_VFWEC"),
        variable = cms.string("pt")
    ),
    cms.PSet(
        category = cms.uint32(15), # minMVAWgWgsfEC
        cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_wGwGSF_FWEC"),
        variable = cms.string("pt")
    ),
    cms.PSet(
        category = cms.uint32(16), # minMVAWgWgsfVFEC
        cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_wGwGSF_VFWEC"),
        variable = cms.string("pt")
    )
)

workingPoints_phase2 = cms.vstring(
    "_WPEff98",
    "_WPEff90",
    "_WPEff80",
    "_WPEff70",
    "_WPEff60"
)
