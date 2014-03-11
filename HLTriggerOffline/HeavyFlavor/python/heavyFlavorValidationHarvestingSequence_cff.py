from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationHarvesting_cfi import *

hfv1 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon0_Jpsi_NoVertexing_v')
)
hfv2 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon0_Jpsi_v')
)
hfv3 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Dimuon10_Jpsi_v')
)
hfv4 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu5_L2Mu3_Jpsi_v')
)
hfv5 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu5_Track2_Jpsi_v')
)
hfv6 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu7_Track7_Jpsi_v')
)
hfv7 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_DoubleMu4_Jpsi_Displaced_v')
)
hfv8 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu3')
)
hfv9 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_DoubleMu3')
)

combiner = cms.EDAnalyzer('PlotCombiner',
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT'),
  Plots = cms.untracked.VPSet(
    cms.untracked.PSet(
      InputMEnames = cms.untracked.vstring(
      	'HLT_Dimuon0_Jpsi_NoVertexing_v/effPathDiglobAND_recoRapPtY',
      	'HLT_Dimuon0_Jpsi_v/effPathDiglobAND_recoRapPtY',
      	'HLT_Dimuon10_Jpsi_v/effPathDiglobAND_recoRapPtY',
      	'HLT_Mu5_L2Mu3_Jpsi_v/effPathDiglobAND_recoRapPtY',
      	'HLT_Mu5_Track2_Jpsi_v/effPathDiglobAND_recoRapPtY',
      	'HLT_Mu7_Track7_Jpsi_v/effPathDiglobAND_recoRapPtY',
        'HLT_DoubleMu4_Jpsi_Displaced_v/effPathDiglobAND_recoRapPtY',
        'HLT_Mu3/effPathDiglobAND_recoRapPtY',
        'HLT_DoubleMu3/effPathDiglobAND_recoRapPtY',
      ),
      InputLabels = cms.untracked.vstring(
        'HLT_Dimuon0_Jpsi_NoVertexing_v',
        'HLT_Dimuon0_Jpsi_v',
        'HLT_Dimuon10_Jpsi_v',
        'HLT_Mu5_L2Mu3_Jpsi_v',
        'HLT_Mu5_Track2_Jpsi_v',
        'HLT_Mu7_Track7_Jpsi_v',
        'HLT_DoubleMu4_Jpsi_Displaced_v',
        'HLT_Mu3',
        'HLT_DoubleMu3',
      ),
      OutputMEname = cms.untracked.string('effPathGlob_recoPt')
    )
  )
)

heavyFlavorValidationHarvestingSequence = cms.Sequence(hfv1+hfv2+hfv3+hfv4+hfv5+hfv6+hfv7
													  +hfv8+hfv9
													  +combiner)

