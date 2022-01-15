import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tdeStage2RegionalShowerClient = DQMEDHarvester(
    "L1TdeStage2RegionalShowerClient",
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2EMTF/Shower"),
)

