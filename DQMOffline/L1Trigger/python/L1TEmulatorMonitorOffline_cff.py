import FWCore.ParameterSet.Config as cms

# adapt the L1TEmulatorMonitor_cff configuration to offline DQM

# DQM online L1 Trigger emulator modules 
from DQM.L1TMonitor.L1TEmulatorMonitor_cff import *


# Stage 2

from DQM.L1TMonitor.L1TStage2Emulator_cff import *
# add calo layer 2 emulation with inputs from the calo layer 1 emulator since the full unpacked data to emulate layer 2 is only available for validation events
valCaloStage2Layer2DigisOffline = valCaloStage2Layer2Digis.clone()
valCaloStage2Layer2DigisOffline.towerToken = cms.InputTag("valCaloStage2Layer1Digis")

Stage2L1HardwareValidationForOfflineCalo = cms.Sequence(valCaloStage2Layer2DigisOffline)

# Calo layer 2 emulator DQM modules for offline
from DQM.L1TMonitor.L1TdeStage2CaloLayer2_cfi import *
l1tdeStage2CaloLayer2Offline = l1tdeStage2CaloLayer2.clone()
l1tdeStage2CaloLayer2Offline.calol2JetCollectionEmul = cms.InputTag("valCaloStage2Layer2DigisOffline")
l1tdeStage2CaloLayer2Offline.calol2EGammaCollectionEmul = cms.InputTag("valCaloStage2Layer2DigisOffline")
l1tdeStage2CaloLayer2Offline.calol2TauCollectionEmul = cms.InputTag("valCaloStage2Layer2DigisOffline")
l1tdeStage2CaloLayer2Offline.calol2EtSumCollectionEmul = cms.InputTag("valCaloStage2Layer2DigisOffline")

from DQM.L1TMonitor.L1TStage2CaloLayer2Emul_cfi import *
l1tStage2CaloLayer2EmulOffline = l1tStage2CaloLayer2Emul.clone()
l1tStage2CaloLayer2EmulOffline.stage2CaloLayer2JetSource = cms.InputTag("valCaloStage2Layer2DigisOffline")
l1tStage2CaloLayer2EmulOffline.stage2CaloLayer2EGammaSource = cms.InputTag("valCaloStage2Layer2DigisOffline")
l1tStage2CaloLayer2EmulOffline.stage2CaloLayer2TauSource = cms.InputTag("valCaloStage2Layer2DigisOffline")
l1tStage2CaloLayer2EmulOffline.stage2CaloLayer2EtSumSource = cms.InputTag("valCaloStage2Layer2DigisOffline")

l1tStage2EmulatorOfflineDQMForCalo = cms.Sequence(
    l1tdeStage2CaloLayer2Offline +
    l1tStage2CaloLayer2EmulOffline
)

