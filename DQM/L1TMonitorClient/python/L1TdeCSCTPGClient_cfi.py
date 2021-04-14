import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.L1TMonitor.L1TdeCSCTPG_cfi import l1tdeCSCTPGCommon

l1tdeCSCTPGClient = DQMEDHarvester(
    "L1TdeCSCTPGClient",
    l1tdeCSCTPGCommon
)
