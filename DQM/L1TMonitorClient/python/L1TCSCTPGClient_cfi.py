import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tCSCTPGClient = DQMEDHarvester("L1TCSCTPGClient",
    monitorDir = cms.string('L1TEMU/L1TdeCSCTPG'),
)
