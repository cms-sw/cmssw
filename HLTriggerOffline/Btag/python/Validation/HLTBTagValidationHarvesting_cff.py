from HLTriggerOffline.Btag.Validation.HLTBTagHarvestingAnalyzer_cff import *



bTagharvest=bTagPostValidation.clone()


HLTBTagHarvestingSequence = cms.Sequence(bTagharvest)

