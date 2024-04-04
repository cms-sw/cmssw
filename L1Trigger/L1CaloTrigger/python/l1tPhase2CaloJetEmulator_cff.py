from L1Trigger.L1CaloTrigger.l1tPhase2CaloJetEmulator_cfi import *

from L1Trigger.L1THGCal.l1tHGCalTowerMapProducer_cfi import *
from L1Trigger.L1THGCal.l1tHGCalTowerMapProducer_cfi import L1TTriggerTowerConfig_energySplit
from L1Trigger.L1THGCal.l1tHGCalTowerProducer_cfi import *

# Add HGCal tower producers for energy split towers
# Based on custom_towers_energySplit in L1Trigger/L1THGCal/python/customTowers.py
parameters_towers_2d = L1TTriggerTowerConfig_energySplit.clone()
l1tHGCalEnergySplitTowerMapProducer = l1tHGCalTowerMapProducer.clone()
l1tHGCalEnergySplitTowerMapProducer.ProcessorParameters.towermap_parameters.L1TTriggerTowerConfig = parameters_towers_2d
l1tHGCalEnergySplitTowerProducer = l1tHGCalTowerProducer.clone( InputTowerMaps = ("l1tHGCalEnergySplitTowerMapProducer","HGCalTowerMapProcessor") )
l1tHGCalEnergySplitTowersTask = cms.Task(l1tHGCalEnergySplitTowerMapProducer, l1tHGCalEnergySplitTowerProducer)

# Use energy split towers in calo jet/tau emulator
l1tPhase2CaloJetEmulator.hgcalTowers = ("l1tHGCalEnergySplitTowerProducer","HGCalTowerProcessor")

l1tCaloJetsTausTask = cms.Task(
    l1tHGCalEnergySplitTowersTask,
    l1tPhase2CaloJetEmulator
)
