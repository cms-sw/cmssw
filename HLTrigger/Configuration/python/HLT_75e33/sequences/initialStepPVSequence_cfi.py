import FWCore.ParameterSet.Config as cms

from ..tasks.initialStepPVTask_cfi import *

initialStepPVSequence = cms.Sequence(initialStepPVTask)
