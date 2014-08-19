from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationHarvesting_cfi import *

hfv1 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_DoubleMu4_3_Bs_v')
)
hfv2 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_DoubleMu4_3_Jpsi_Displaced_v')
)
hfv3 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon20_Jpsi_v')
)
hfv4 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon13_PsiPrime_v')
)
hfv5 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon13_Upsilon_v')
)
hfv6 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu25_TkMu0_dEta18_Onia_v')
)
hfv7 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_DoubleMu4_JpsiTrk_Displaced_v')
)
hfv8 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_DoubleMu4_PsiPrimeTrk_Displaced_v')
)
hfv9 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_DoubleMu4_LowMassNonResonantTrk_Displaced_v')
)

combiner = cms.EDAnalyzer('PlotCombiner',
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT'),
  Plots = cms.untracked.VPSet(
    cms.untracked.PSet(
      InputMEnames = cms.untracked.vstring(
        'HLT_DoubleMu4_3_Bs_v/effPathDiglobAND_recoRapPtY',
        'HLT_DoubleMu4_3_Jpsi_Displaced_v/effPathDiglobAND_recoRapPtY',
        'HLT_Dimuon20_Jpsi_v/effPathDiglobAND_recoRapPtY',
        'HLT_Dimuon13_PsiPrime_v/effPathDiglobAND_recoRapPtY',
        'HLT_Dimuon13_Upsilon_v/effPathDiglobAND_recoRapPtY',
        'HLT_Mu25_TkMu0_dEta18_Onia_v/effPathDiglobAND_recoRapPtY',
        'HLT_DoubleMu4_JpsiTrk_Displaced_v/effPathDiglobAND_recoRapPtY',
        'HLT_DoubleMu4_PsiPrimeTrk_Displaced_v/effPathDiglobAND_recoRapPtY',
        'HLT_DoubleMu4_LowMassNonResonantTrk_Displaced_v/effPathDiglobAND_recoRapPtY',
      ),
      InputLabels = cms.untracked.vstring(
        'HLT_DoubleMu4_3_Bs_v',
        'HLT_DoubleMu4_3_Jpsi_Displaced_v',
        'HLT_Dimuon20_Jpsi_v',
        'HLT_Dimuon13_PsiPrime_v',
        'HLT_Dimuon13_Upsilon_v',
        'HLT_Mu25_TkMu0_dEta18_Onia_v',
        'HLT_DoubleMu4_JpsiTrk_Displaced_v',
        'HLT_DoubleMu4_PsiPrimeTrk_Displaced_v',
        'HLT_DoubleMu4_LowMassNonResonantTrk_Displaced_v',
      ),
      OutputMEname = cms.untracked.string('effPathGlob_recoPt')
    )
  )
)

heavyFlavorValidationHarvestingSequence = cms.Sequence(hfv1+hfv2+hfv3+hfv4+hfv5+hfv6+hfv7+hfv8+hfv9
													  +combiner)

