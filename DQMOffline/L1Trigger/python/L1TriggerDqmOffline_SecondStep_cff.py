import FWCore.ParameterSet.Config as cms

from DQMOffline.L1Trigger.L1TStage2CaloLayer2Efficiency_cfi import *
from DQMOffline.L1Trigger.L1TStage2CaloLayer2Diff_cfi import *

# l1tStage2CaloLayer2EmuDiff uses plots produced by
# l1tStage2CaloLayer2Efficiency
DQMHarvestL1Trigger = cms.Sequence(
    l1tStage2CaloLayer2Efficiency * l1tStage2CaloLayer2EmuDiff
)

