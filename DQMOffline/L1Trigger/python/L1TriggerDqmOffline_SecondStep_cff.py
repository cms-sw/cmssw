import FWCore.ParameterSet.Config as cms

from DQMOffline.L1Trigger.L1TStage2CaloLayer2Efficiency_cfi import *
from DQMOffline.L1Trigger.L1TStage2CaloLayer2Diff_cfi import *

from DQMOffline.L1Trigger.L1TEGammaEfficiency_cfi import *
from DQMOffline.L1Trigger.L1TEGammaDiff_cfi import *

from DQMOffline.L1Trigger.L1TTauEfficiency_cfi import *
from DQMOffline.L1Trigger.L1TTauDiff_cfi import *
from DQMOffline.L1Trigger.L1TMuonDQMEfficiency_cff import *

# harvesting sequence for electron dataset
DQMHarvestL1TEg = cms.Sequence(
    l1tEGammaEfficiency
    #* l1tEGammaEmuEfficiency
    #* l1tEGammaEmuDiff
)

# l1tStage2CaloLayer2EmuDiff uses plots produced by
# l1tStage2CaloLayer2Efficiency

# harvesting sequence for muon dataset
DQMHarvestL1TMuon = cms.Sequence(
    l1tStage2CaloLayer2Efficiency
    #* l1tStage2CaloLayer2EmuEfficiency
    #* l1tStage2CaloLayer2EmuDiff
    * l1tTauEfficiency
    #* l1tTauEmuDiff
    * l1tMuonDQMEfficiency
)

