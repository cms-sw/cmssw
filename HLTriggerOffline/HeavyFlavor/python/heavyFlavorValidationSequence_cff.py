from HLTriggerOffline.HeavyFlavor.heavyFlavorValidation_cfi import *

hfv1 = heavyFlavorValidation.clone(
    #TriggerProcessName = cms.untracked.string("HLT"),
    TriggerPathName = cms.untracked.string("HLT_Dimuon0_Jpsi_NoVertexing_v"),
    MuonPtBins = cms.untracked.vdouble(0.5, 1., 2., 3., 3.5, 4., 5., 6., 7., 8., 9., 10., 20.),
    MuonEtaBins = cms.untracked.vdouble(24, -2.4, 2.4),
    DimuonPtBins = cms.untracked.vdouble(0., 2., 4., 6., 8., 10., 12., 15., 25.)
)
hfv2 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon0_Jpsi_v")
)
hfv3 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Dimuon10_Jpsi_v")
)
hfv4 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu5_L2Mu3_Jpsi_v")
)
hfv5 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu5_Track2_Jpsi_v")
)
hfv6 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu7_Track7_Jpsi_v")
)
hfv7 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_DoubleMu4_Jpsi_Displaced_v")
)
hfv8 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu3")
)
hfv9 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_DoubleMu3")
)

heavyFlavorValidationSequence = cms.Sequence(
  hfv1+hfv2+hfv3+hfv4+hfv5+hfv6+hfv7
  +hfv8+hfv9
)
