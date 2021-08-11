import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.L1TMonitor.L1TdeCSCTPG_cfi import l1tdeCSCTPGCommon

l1tdeCSCTPGClient = DQMEDHarvester(
    "L1TdeCSCTPGClient",
    l1tdeCSCTPGCommon
)

# enable comparisons for Run-3 data members
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify( l1tdeCSCTPGClient,
                      isRun3 = True)
