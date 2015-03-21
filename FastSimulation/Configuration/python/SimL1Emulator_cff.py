import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.SimL1Emulator_cff import *

for _entry in [SimL1TechnicalTriggers]:
    SimL1Emulator.remove(_entry)

del simGtDigis.TechnicalTriggersInputTags

