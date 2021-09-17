import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.L1TMonitor.L1TdeGEMTPG_cfi import l1tdeGEMTPGCommon

l1tdeGEMTPGClient = DQMEDHarvester(
    "L1TdeGEMTPGClient",
    l1tdeGEMTPGCommon
)
