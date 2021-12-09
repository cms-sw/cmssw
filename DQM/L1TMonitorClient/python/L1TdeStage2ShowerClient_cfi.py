import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tdeStage2ShowerClient = DQMEDHarvester(
    "L1TdeStage2ShowerClient",
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2EMTF/Shower"),
)

