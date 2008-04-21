import FWCore.ParameterSet.Config as cms

# -*-SH-*-
MuonCaloCompatibilityBlock = cms.PSet(
    MuonCaloCompatibility = cms.PSet(
        PionTemplateFileName = cms.FileInPath('RecoMuon/MuonIdentification/data/MuID_templates_pions_allPt_2_0_norm.root'),
        MuonTemplateFileName = cms.FileInPath('RecoMuon/MuonIdentification/data/MuID_templates_muons_allPt_2_0_norm.root')
    )
)

