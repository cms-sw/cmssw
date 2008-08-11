import FWCore.ParameterSet.Config as cms

# this file is still used by FastSim
# no-one else should use it!!!

# Configuration (use Fake/Frontier in future)
from L1Trigger.Configuration.L1Config_cff import *
# Emulator modules
from L1Trigger.Configuration.L1MuonEmulator_cff import *
from L1Trigger.Configuration.L1CaloEmulator_cff import *
from L1Trigger.GlobalTrigger.gtDigis_cfi import *
# Emulator sequence
L1Emulator = cms.Sequence(L1CaloEmulator*L1MuonEmulator*gtDigis)

