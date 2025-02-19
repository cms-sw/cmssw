from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationHarvesting_cfi import *

hfv1 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu3')
)
hfv2 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_IsoMu3')
)
hfv3 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu5')
)
hfv4 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu9')
)
hfv5 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_DoubleMu0')
)
hfv6 = heavyFlavorValidationHarvesting.clone(
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_DoubleMu3')
)

combiner = cms.EDAnalyzer('PlotCombiner',
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT'),
  Plots = cms.untracked.VPSet(
    cms.untracked.PSet(
      InputMEnames = cms.untracked.vstring('HLT_Mu3/effPathDiglobOR_recoRapPtY','HLT_Mu5/effPathDiglobOR_recoRapPtY','HLT_Mu9/effPathDiglobOR_recoRapPtY','HLT_DoubleMu0/effPathDiglobAND_recoRapPtY','HLT_DoubleMu3/effPathDiglobAND_recoRapPtY'),
      InputLabels = cms.untracked.vstring('HLT_Mu3','HLT_Mu5','HLT_Mu9','HLT_DoubleMu0','HLT_DoubleMu3'),
      OutputMEname = cms.untracked.string('effPathGlob_recoPt')
    )
  )
)

heavyFlavorValidationHarvestingSequence = cms.Sequence(hfv1+hfv2+hfv3+hfv4+hfv5+hfv6+combiner)

