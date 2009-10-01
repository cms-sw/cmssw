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

heavyFlavorValidationHarvestingSequence = cms.Sequence(hfv1+hfv2+hfv3+hfv4+hfv5+hfv6)

