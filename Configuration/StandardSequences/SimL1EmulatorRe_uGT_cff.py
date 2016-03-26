import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

## L1RE FULL:  Re-Emulate all of L1 


if not (eras.stage2L1Trigger.isChosen()):
    print "L1T WARN:  L1RE ALL only supports Stage 2 eras for now."
else:
    print "L1T INFO:  L1RE uGT will re-emulate uGT (Stage-2) using unpacked emulated Calo Layer2 and uGMT."

    from L1Trigger.Configuration.SimL1Emulator_cff import *

    SimL1Emulator = cms.Sequence(SimL1GtEmulatorCore)
