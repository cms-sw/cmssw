import FWCore.ParameterSet.Config as cms

from ..sequences.SimL1EmulatorTask_cfi import *

SimL1Emulator = cms.Sequence(SimL1EmulatorTask)
