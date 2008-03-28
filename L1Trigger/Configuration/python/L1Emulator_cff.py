import FWCore.ParameterSet.Config as cms

# Emulator configuration
# select one of the two lines below
# By default (17X) take configuration from dummy producers
from L1Trigger.Configuration.L1Config_cff import *
# Emulator modules
from L1Trigger.Configuration.L1MuonEmulator_cff import *
from L1Trigger.Configuration.L1CaloEmulator_cff import *
from L1Trigger.GlobalTrigger.gtDigis_cfi import *
from EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi import *
# Emulator sequence
L1Emulator = cms.Sequence(L1CaloEmulator*L1MuonEmulator*gtDigis*l1GtRecord)

