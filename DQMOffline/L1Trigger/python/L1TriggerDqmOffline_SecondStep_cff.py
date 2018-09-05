import FWCore.ParameterSet.Config as cms

from DQMOffline.L1Trigger.L1TMonitorClientOffline_cff import *
from DQMOffline.L1Trigger.L1TEmulatorMonitorClientOffline_cff import *

from DQMOffline.L1Trigger.L1TEtSumEfficiency_cfi import *
from DQMOffline.L1Trigger.L1TEtSumDiff_cfi import *

from DQMOffline.L1Trigger.L1TJetEfficiency_cfi import *
from DQMOffline.L1Trigger.L1TJetDiff_cfi import *

from DQMOffline.L1Trigger.L1TEGammaEfficiency_cfi import *
from DQMOffline.L1Trigger.L1TEGammaDiff_cfi import *

from DQMOffline.L1Trigger.L1TTauEfficiency_cfi import *
from DQMOffline.L1Trigger.L1TTauDiff_cfi import *

from DQMOffline.L1Trigger.L1TMuonDQMEfficiency_cff import *

# harvesting sequence for all datasets
DQMHarvestL1TMon = cms.Sequence(
    l1tStage2MonitorClientOffline
    * l1tStage2EmulatorMonitorClientOffline
)

# harvesting sequence for electron dataset
DQMHarvestL1TEg = cms.Sequence(
    l1tEGammaEfficiency
    * l1tEGammaEmuEfficiency
    #* l1tEGammaEmuDiff
)

# l1tEtSumEmuDiff uses plots produced by
# l1tEtSumEfficiency
# l1tJetEmuDiff uses plots produced by
# l1tJetEfficiency

# harvesting sequence for muon dataset
DQMHarvestL1TMuon = cms.Sequence(
    l1tEtSumEfficiency
    * l1tJetEfficiency
    * l1tEtSumEmuEfficiency
    * l1tJetEmuEfficiency
    #* l1tEtSumEmuDiff
    #* l1tJetEmuDiff
    * l1tTauEfficiency
    * l1tTauEmuEfficiency
    #* l1tTauEmuDiff
    * l1tMuonDQMEfficiency
    * l1tMuonDQMEmuEfficiency
)

