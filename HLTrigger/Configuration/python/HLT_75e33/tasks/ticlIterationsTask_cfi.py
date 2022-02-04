import FWCore.ParameterSet.Config as cms

from ..tasks.ticlEMStepTask_cfi import *
from ..tasks.ticlHADStepTask_cfi import *
from ..tasks.ticlTrkEMStepTask_cfi import *
from ..tasks.ticlTrkStepTask_cfi import *

ticlIterationsTask = cms.Task(ticlEMStepTask, ticlHADStepTask, ticlTrkEMStepTask, ticlTrkStepTask)
