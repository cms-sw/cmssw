import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.MIPStep_cff import *
from RecoHGCal.TICL.TrkStep_cff import *
from RecoHGCal.TICL.EMStep_cff import *
from RecoHGCal.TICL.HADStep_cff import *


iterTICLTask = cms.Task(MIPStepTask,
    TrkStepTask,
    EMStepTask,
    HADStepTask
    )

iterTICL = cms.Sequence(iterTICLTask)

