from HLTriggerOffline.HeavyFlavor.heavyFlavorValidation_cfi import *

hfv1 = heavyFlavorValidation.clone(
    TriggerProcessName = cms.untracked.string("HLT"),
    TriggerPathName = cms.untracked.string("HLT_Mu3"),
    MuonPtBins = cms.untracked.vdouble(1., 2.5, 3.5, 4.5, 6., 8., 10., 20., 50.),
    MuonEtaBins = cms.untracked.vdouble(24, -2.4, 2.4)
)
hfv2 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_IsoMu3")
)
hfv3 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_DoubleMu0")
)
hfv4 = hfv1.clone(
    TriggerPathName = cms.untracked.string("HLT_DoubleMu3")
)

heavyFlavorValidationSequence = cms.Sequence(
  hfv1+hfv2+hfv3+hfv4
)
