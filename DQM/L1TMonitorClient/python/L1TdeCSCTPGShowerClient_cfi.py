import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tdeCSCTPGShowerClient = DQMEDHarvester(
    "L1TdeCSCTPGShowerClient",
    monitorDir = cms.untracked.string("L1TEMU/L1TdeCSCTPGShower"),
)

# foo bar baz
# w48HVgwCMPSEh
# kc49LWHvn8Qxq
