from HLTriggerOffline.HeavyFlavor.heavyFlavorValidation_cfi import *

hfv1 = heavyFlavorValidation.clone(
    TriggerProcessName = cms.untracked.string("HLT"),
    TriggerPathName = cms.untracked.string("HLT_Mu3"),
    MuonPtBins = cms.untracked.vdouble(1., 3., 4., 5., 7., 10., 20., 50.),
    MuonEtaBins = cms.untracked.vdouble(24, -2.4, 2.4)
)
hfv2 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_IsoMu3")
)
hfv3 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu5")
)
hfv4 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_Mu9")
)
hfv5 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_DoubleMu0")
)
hfv6 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_DoubleMu3")
)

heavyFlavorValidationSequence = cms.Sequence(
  hfv1+hfv2+hfv3+hfv4+hfv5+hfv6
)
