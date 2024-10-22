from HLTrigger.HLTcore.hltEventAnalyzerAODDefault_cfi import hltEventAnalyzerAODDefault as _hltEventAnalyzerAODDefault

hltEventAnalyzerAOD = _hltEventAnalyzerAODDefault.clone();

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(hltEventAnalyzerAOD, stageL1Trigger = 2)
