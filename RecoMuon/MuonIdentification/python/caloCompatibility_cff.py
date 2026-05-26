import FWCore.ParameterSet.Config as cms

MuonCaloCompatibilityBlock = cms.PSet(
    MuonCaloCompatibility = cms.PSet(
        PionTemplateFileName = cms.FileInPath('RecoMuon/MuonIdentification/data/MuID_templates_pions_lowPt_3_1_norm.root'),
        MuonTemplateFileName = cms.FileInPath('RecoMuon/MuonIdentification/data/MuID_templates_muons_lowPt_3_1_norm.root'),
        delta_eta = cms.double(0.02),
        delta_phi = cms.double(0.02),
        allSiPMHO = cms.bool(False),
        isPhase2 = cms.bool(False), #for Phase-1
    )
)

from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0
phase2_GE0.toModify(MuonCaloCompatibilityBlock.MuonCaloCompatibility,
                    isPhase2 = True)


