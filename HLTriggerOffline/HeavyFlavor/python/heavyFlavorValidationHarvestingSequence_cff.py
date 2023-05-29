from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationHarvesting_cfi import *
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

baseFolderPath = 'HLT/BPH/HLT/'

hfv1 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_DoubleMu4_3_Bs_v')
)
hfv2 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_DoubleMu4_3_Jpsi_v')
)


hfvTnP1 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Mu7p5_L2Mu2_Jpsi_v')
)
hfvTnP2 = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Mu7p5_L2Mu2_Upsilon_v')
)

hfv7 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_DoubleMu4_MuMuTrk_Displaced_v')
)

### 2017 additions
hfjpsiMuon = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Dimuon0_Jpsi3p5_Muon2_v')
)

hfjpsiNoVertexing = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Dimuon0_Jpsi_NoVertexing_v')
)

hfjpsiNoVertexingNoOS = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v')
)

hfjpsiBarrel = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Dimuon20_Jpsi_Barrel_Seagulls_v')
)

hfpsiBarrel = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Dimuon10_PsiPrime_Barrel_Seagulls_v')
)

hfupsilonBarrel = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Dimuon10_Upsilon_Barrel_Seagulls_v')
)

hfphiBarrel = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Dimuon10_Phi_Barrel_Seagulls_v')
)

hfmu25Onia = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Mu25_TkMu0_Onia_v')
)

hfmu30Onia = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Mu30_TkMu0_Onia_v')
)

hfmu25Phi = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Mu25_TkMu0_Phi_v')
)

hfmu20Phi = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Mu20_TkMu0_Phi_v')
)

hfjpsi = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Dimuon25_Jpsi_v')
)

hfpsi = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Dimuon14_PsiPrime_v')
)

hfupsilon = heavyFlavorValidationHarvesting.clone(
    MyDQMrootFolder = cms.untracked.string(baseFolderPath + 'HLT_Dimuon10_Upsilon_y1p4_v')
)


combiner = DQMEDHarvester('PlotCombiner',
  MyDQMrootFolder = cms.untracked.string('HLT/BPH/HLT'),
  Plots = cms.untracked.VPSet(
    cms.untracked.PSet(
      InputMEnames = cms.untracked.vstring(
        'HLT_DoubleMu4_3_Bs_v/effPathDiglobAND_recoEtaPtY',
        'HLT_DoubleMu4_3_Jpsi_v/effPathDiglobAND_recoEtaPtY',
        'HLT_DoubleMu4_MuMuTrk_Displaced_v/effPathDiglobAND_recoEtaPtY',
      ),
      InputLabels = cms.untracked.vstring(
        'HLT_DoubleMu4_3_Bs_v',
        'HLT_DoubleMu4_3_Jpsi_v',
        'HLT_DoubleMu4_MuMuTrk_Displaced_v',
      ),
      OutputMEname = cms.untracked.string('effPathGlob_recoLowPt')
    )
  )
)

combiner2 = combiner.clone(
  Plots = cms.untracked.VPSet(
    cms.untracked.PSet(
      InputMEnames = cms.untracked.vstring(
          'HLT_Dimuon25_Jpsi_v/effPathDiglobAND_recoEtaPtY',
          'HLT_Dimuon14_PsiPrime_v/effPathDiglobAND_recoEtaPtY',
          'HLT_Dimuon10_Upsilon_y1p4_v/effPathDiglobAND_recoEtaPtY',
          'HLT_Mu30_TkMu0_Onia_v/effPathDiglobAND_recoEtaPtY'
      ),
      InputLabels = cms.untracked.vstring(
          'HLT_Dimuon25_Jpsi_v',
          'HLT_Dimuon14_PsiPrime_v',
          'HLT_Dimuon10_Upsilon_y1p4_v',
          'HLT_Mu30_TkMu0_Onia_v'
      ),
      OutputMEname = cms.untracked.string('effPathGlob_recoHighPt')
    )
  )
)

heavyFlavorValidationHarvestingSequence = cms.Sequence(
  hfv1+hfv2
  +hfvTnP1+hfvTnP2
  +hfv7
    + hfjpsiMuon
    + hfjpsiNoVertexing
    + hfjpsiNoVertexingNoOS
    + hfjpsiBarrel
    + hfpsiBarrel
    + hfmu25Onia
    + hfmu30Onia
    + hfmu25Phi
    + hfmu20Phi
    + hfjpsi
    + hfpsi
    + hfupsilon
    +combiner+combiner2
)
