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

hfvQuadmu1 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon0_Jpsi_Muon_v')
)
hfvQuadmu2 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon0_Upsilon_Muon_v')
)
hfvQuadmu3 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_QuadMuon0_Dimuon0_Jpsi_v')
)
hfvQuadmu4 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_QuadMuon0_Dimuon0_Upsilon_v')
)

hfvTnP1 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu7p5_L2Mu2_Jpsi_v')
)
hfvTnP2 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu7p5_L2Mu2_Upsilon_v')
)
hfvTnP3 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu7p5_Track2_Jpsi_v')
)
hfvTnP4 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu7p5_Track3p5_Jpsi_v')
)
hfvTnP5 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu7p5_Track7_Jpsi_v')
)
hfvTnP6 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu7p5_Track2_Upsilon_v')
)
hfvTnP7 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu7p5_Track3p5_Upsilon_v')
)
hfvTnP8 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu7p5_Track7_Upsilon_v')
)
hfvTnP9 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon0er16_Jpsi_NoOS_NoVertexing_v')
)
hfvTnP10 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon0er16_Jpsi_NoVertexing_v')
)
hfvTnP11 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon6_Jpsi_NoVertexing_v')
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

### 7E33 quarkonia
hfvQ1 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon10_Jpsi_Barrel_v')
)
hfvQ2 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon16_Jpsi_v')
)
hfvQ3 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon8_PsiPrime_Barrel_v')
)
hfvQ4 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon8_Upsilon_Barrel_v')
)
hfvQ5 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon0_Phi_Barrel_v')
)
hfvQ6 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu16_TkMu0_dEta18_Onia_v')
)
hfvQ7 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu16_TkMu0_dEta18_Phi_v')
)


combiner = cms.EDAnalyzer('PlotCombiner',
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT'),
  Plots = cms.untracked.VPSet(
    cms.untracked.PSet(
      InputMEnames = cms.untracked.vstring(
        'HLT_DoubleMu4_3_Bs_v/effPathDiglobAND_recoEtaPtY',
        'HLT_DoubleMu4_3_Jpsi_Displaced_v/effPathDiglobAND_recoEtaPtY',
        'HLT_DoubleMu4_JpsiTrk_Displaced_v/effPathDiglobAND_recoEtaPtY',
        'HLT_DoubleMu4_PsiPrimeTrk_Displaced_v/effPathDiglobAND_recoEtaPtY',
        'HLT_DoubleMu4_LowMassNonResonantTrk_Displaced_v/effPathDiglobAND_recoEtaPtY',
      ),
      InputLabels = cms.untracked.vstring(
        'HLT_DoubleMu4_3_Bs_v',
        'HLT_DoubleMu4_3_Jpsi_Displaced_v',
        'HLT_DoubleMu4_JpsiTrk_Displaced_v',
        'HLT_DoubleMu4_PsiPrimeTrk_Displaced_v',
        'HLT_DoubleMu4_LowMassNonResonantTrk_Displaced_v',
      ),
      OutputMEname = cms.untracked.string('effPathGlob_recoLowPt')
    )
  )
)

combiner2 = combiner.clone(
  Plots = cms.untracked.VPSet(
    cms.untracked.PSet(
      InputMEnames = cms.untracked.vstring(
        'HLT_Dimuon20_Jpsi_v/effPathDiglobAND_recoEtaPtY',
        'HLT_Dimuon13_PsiPrime_v/effPathDiglobAND_recoEtaPtY',
        'HLT_Dimuon13_Upsilon_v/effPathDiglobAND_recoEtaPtY',
        'HLT_Mu25_TkMu0_dEta18_Onia_v/effPathDiglobAND_recoEtaPtY',
      ),
      InputLabels = cms.untracked.vstring(
        'HLT_Dimuon20_Jpsi_v',
        'HLT_Dimuon13_PsiPrime_v',
        'HLT_Dimuon13_Upsilon_v',
        'HLT_Mu25_TkMu0_dEta18_Onia_v',
      ),
      OutputMEname = cms.untracked.string('effPathGlob_recoHighPt')
    )
  )
)

heavyFlavorValidationHarvestingSequence = cms.Sequence(
  hfv1+hfv2+hfv3+hfv4+hfv5 
  +hfvQuadmu1+hfvQuadmu2+hfvQuadmu3+hfvQuadmu4
  +hfvTnP1+hfvTnP2+hfvTnP3+hfvTnP4+hfvTnP5+hfvTnP6+hfvTnP7+hfvTnP8+hfvTnP9+hfvTnP10+hfvTnP11
  +hfv6+hfv7+hfv8+hfv9
  +hfvQ1+hfvQ2+hfvQ3+hfvQ4+hfvQ5+hfvQ6+hfvQ7
	+combiner+combiner2
)

