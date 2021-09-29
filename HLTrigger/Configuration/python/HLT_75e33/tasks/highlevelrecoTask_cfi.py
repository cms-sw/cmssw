import FWCore.ParameterSet.Config as cms

from ..tasks.particleFlowRecoTask_cfi import *

highlevelrecoTask = cms.Task(
    particleFlowRecoTask
)
