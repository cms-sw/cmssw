import FWCore.ParameterSet.Config as cms

from ..sequences.SimL1Emulator_cfi import *

L1simulation_step = cms.Path(SimL1Emulator)
