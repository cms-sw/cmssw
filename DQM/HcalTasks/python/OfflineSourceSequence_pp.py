import FWCore.ParameterSet.Config as cms

#-----------------
#   HCAL DQM Offline Source Sequence Definition for pp
#   To be used for Offline DQM importing
#-----------------

#   import the tasks
from DQM.HcalTasks.DigiTask import digiTask
from DQM.HcalTasks.RawTask import rawTask
from DQM.HcalTasks.TPTask import tpTask
from DQM.HcalTasks.RecHitTask import recHitTask, recHitPreRecoTask
from DQM.HcalTasks.hcalGPUComparisonTask_cfi import hcalGPUComparisonTask

#   set processing type to Offine
digiTask.ptype = 1
tpTask.ptype = 1
recHitTask.ptype = 1
rawTask.ptype = 1
recHitPreRecoTask.ptype = 1
hcalGPUComparisonTask.ptype = 1

#   set the label for Emulator TP Task
tpTask.tagEmul = "valHcalTriggerPrimitiveDigis"

hcalOfflineSourceSequence = cms.Sequence(
    digiTask +
    tpTask +
    recHitTask +
    rawTask )

hcalOnlyOfflineSourceSequence = cms.Sequence(
    digiTask +
    recHitPreRecoTask +
    rawTask )

hcalOnlyOfflineSourceSequenceGPU = cms.Sequence(
    digiTask +
    recHitTask +
    rawTask +
    hcalGPUComparisonTask
)

from Configuration.ProcessModifiers.gpuValidationHcal_cff import gpuValidationHcal
gpuValidationHcal.toReplaceWith(hcalOnlyOfflineSourceSequence, hcalOnlyOfflineSourceSequenceGPU)

from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018
run2_HCAL_2018.toModify(hcalGPUComparisonTask,
    tagHBHE_ref = "hbheprereco@cpu",
    tagHBHE_target = "hbheprereco@cuda"
)
run2_HCAL_2018.toModify(recHitTask,
    tagHBHE = "hbheprereco"
)

from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
### reverting the reco tag setting that inherited from run2
run3_HB.toModify(hcalGPUComparisonTask,
    tagHBHE_ref = "hbhereco@cpu",
    tagHBHE_target = "hbhereco@cuda"
)
run3_HB.toModify(recHitTask,
    tagHBHE = "hbhereco"
)
_phase1_hcalOnlyOfflineSourceSequence = hcalOnlyOfflineSourceSequence.copy()
_phase1_hcalOnlyOfflineSourceSequence.replace(recHitPreRecoTask, recHitTask)
run3_HB.toReplaceWith(hcalOnlyOfflineSourceSequence, _phase1_hcalOnlyOfflineSourceSequence)

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
_phase2_hcalOfflineSourceSequence = hcalOfflineSourceSequence.copyAndExclude([tpTask,rawTask])
phase2_hcal.toReplaceWith(hcalOfflineSourceSequence, _phase2_hcalOfflineSourceSequence)
phase2_hcal.toModify(digiTask,
    tagHBHE = "simHcalDigis:HBHEQIE11DigiCollection",
    tagHO = "simHcalDigis",
    tagHF = "simHcalDigis:HFQIE10DigiCollection"
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
(premix_stage2 & phase2_hcal).toModify(digiTask,
    tagHBHE = "DMHcalDigis:HBHEQIE11DigiCollection",
    tagHO = "DMHcalDigis",
    tagHF = "DMHcalDigis:HFQIE10DigiCollection"
)
