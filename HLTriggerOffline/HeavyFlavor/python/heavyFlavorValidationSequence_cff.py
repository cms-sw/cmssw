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

hfvQuadmu1 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon0_Jpsi_Muon_v")
)
hfvQuadmu2 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon0_Upsilon_Muon_v")
)
hfvQuadmu3 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_QuadMuon0_Dimuon0_Jpsi_v")
)
hfvQuadmu4 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_QuadMuon0_Dimuon0_Upsilon_v")
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
hfvTnP9 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon0er16_Jpsi_NoOS_NoVertexing_v")
)
hfvTnP10 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon0er16_Jpsi_NoVertexing_v")
)
hfvTnP11 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon6_Jpsi_NoVertexing_v")
)

### Quarkonia 7E33
hfvQ1 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon10_Jpsi_Barrel_v")
)
hfvQ2 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon16_Jpsi_v")
)
hfvQ3 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon8_PsiPrime_Barrel_v")
)
hfvQ4 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon8_Upsilon_Barrel_v")
)
hfvQ5 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon0_Phi_Barrel_v")
)
hfvQ6 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu16_TkMu0_dEta18_Onia_v")
)
hfvQ7 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu16_TkMu0_dEta18_Phi_v")
)

hfv6 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon20_Jpsi_v"),
    MuonPtBins = cms.untracked.vdouble(7.5, 10., 12.5, 15., 17.5, 20., 22.5, 25., 27.5, 30.),
    DimuonPtBins = cms.untracked.vdouble(10., 12.5, 15., 17.5, 20., 22.5, 25., 27.5, 30.)
)
hfv7 = hfv6.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon13_PsiPrime_v"),
)
hfv8 = hfv6.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon13_Upsilon_v"),
)
hfv9 = hfv6.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu25_TkMu0_dEta18_Onia_v"),
)


heavyFlavorValidationSequence = cms.Sequence(
  hfv1+hfv2+hfv3+hfv4+hfv5 
  +hfvQuadmu1+hfvQuadmu2+hfvQuadmu3+hfvQuadmu4
  +hfvTnP1+hfvTnP2+hfvTnP3+hfvTnP4+hfvTnP5+hfvTnP6+hfvTnP7+hfvTnP8+hfvTnP9+hfvTnP10+hfvTnP11
  +hfv6+hfv7+hfv8+hfv9
  +hfvQ1+hfvQ2+hfvQ3+hfvQ4+hfvQ5+hfvQ6+hfvQ7
)
