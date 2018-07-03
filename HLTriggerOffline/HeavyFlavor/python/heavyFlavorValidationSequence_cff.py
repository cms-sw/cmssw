from HLTriggerOffline.HeavyFlavor.heavyFlavorValidation_cfi import *

hfv1 = heavyFlavorValidation.clone(
    #TriggerProcessName = cms.untracked.string("HLT"),
    TriggerPathName = cms.untracked.string("HLT_DoubleMu4_3_Bs_v"),
    MuonPtBins = cms.untracked.vdouble(0.5, 1., 2., 3., 3.5, 4., 5., 6., 7., 8., 9., 10., 20.),
    MuonEtaBins = cms.untracked.vdouble(24, -2.4, 2.4),
    DimuonPtBins = cms.untracked.vdouble(0., 2., 4., 6., 8., 10., 12., 15., 25.)
)
hfv2 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_DoubleMu4_3_Jpsi_Displaced_v")
)
hfv3 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_DoubleMu4_JpsiTrk_Displaced_v")
)
hfv4 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_DoubleMu4_PsiPrimeTrk_Displaced_v")
)
hfv5 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_DoubleMu4_LowMassNonResonantTrk_Displaced_v")
)

hfvTnP1 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu7p5_L2Mu2_Jpsi_v")
)
hfvTnP2 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu7p5_L2Mu2_Upsilon_v")
)
hfvTnP3 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu7p5_Track2_Jpsi_v")
)
hfvTnP4 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu7p5_Track3p5_Jpsi_v")
)
hfvTnP5 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu7p5_Track7_Jpsi_v")
)
hfvTnP6 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu7p5_Track2_Upsilon_v")
)
hfvTnP7 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu7p5_Track3p5_Upsilon_v")
)
hfvTnP8 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu7p5_Track7_Upsilon_v")
)

## 2017 additions
hfjpsiMuon = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon0_Jpsi3p5_Muon2_v")
)

hfjpsiNoVertexing = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon0_Jpsi_NoVertexing_v")
)

hfjpsiNoVertexingNoOS = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v")
)

hfjpsiBarrel = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon20_Jpsi_Barrel_Seagulls_v")
)

hfpsiBarrel = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon10_PsiPrime_Barrel_Seagulls_v")
)

hfupsilonBarrel = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon10_Upsilon_Barrel_Seagulls_v")
)

hfphiBarrel = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon14_Phi_Barrel_Seagulls_v")
)

hfmu25Onia = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu25_TkMu0_Onia_v")
)

hfmu30Onia = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu30_TkMu0_Onia_v")
)

hfmu25Phi = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu25_TkMu0_Phi_v")
)

hfmu20Phi = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu20_TkMu0_Phi_v")
)

hfjpsi = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon25_Jpsi_v")
)

hfpsi = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon18_PsiPrime_v")
)

hfupsilon = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon12_Upsilon_eta1p5_v")
)


heavyFlavorValidationSequence = cms.Sequence(
    hfv1+hfv2+hfv3+hfv4+hfv5
    +hfvTnP1+hfvTnP2+hfvTnP3+hfvTnP4+hfvTnP5+hfvTnP6+hfvTnP7+hfvTnP8
    + hfjpsiMuon
    + hfjpsiNoVertexing
    + hfjpsiNoVertexingNoOS
    + hfjpsiBarrel
    + hfpsiBarrel
    + hfupsilonBarrel
    + hfphiBarrel
    + hfmu25Onia
    + hfmu30Onia
    + hfmu25Phi
    + hfmu20Phi
    + hfjpsi
    + hfpsi
    + hfupsilon
)
