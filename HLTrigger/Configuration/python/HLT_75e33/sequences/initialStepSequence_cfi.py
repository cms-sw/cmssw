import FWCore.ParameterSet.Config as cms

from ..tasks.initialStepTask_cfi import *

initialStepSequence = cms.Sequence(initialStepTask)
