from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger

if not stage1L1Trigger.isChosen() and not stage2L1Trigger.isChosen():
    from DQMOffline.L1Trigger.LegacyL1TriggerDqmOffline_cff import * #LEGACY
else:
    from DQMOffline.L1Trigger.Stage2L1TriggerDqmOffline_cff import * #UPGRADE
