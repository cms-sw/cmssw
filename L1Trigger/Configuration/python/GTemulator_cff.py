import FWCore.ParameterSet.Config as cms

# Global Trigger
from L1Trigger.Phase2L1GT.l1tGTProducer_cff import *
from L1Trigger.Phase2L1GT.l1tGTAlgoBlockProducer_cff import *

# define a core which can be extented in customizations:
GTemulatorTask = cms.Task(
    l1tGTProducer,
    l1tGTAlgoBlockProducer
)

GTemulator = cms.Sequence( GTemulatorTask )
