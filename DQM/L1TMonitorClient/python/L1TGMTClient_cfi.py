import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tGmtClient = DQMEDHarvester("L1TGMTClient",
    input_dir = cms.untracked.string('L1T/L1TGMT'),
    monitorName = cms.untracked.string('L1T/L1TGMT'),
    output_dir = cms.untracked.string('L1T/L1TGMT/Client'),
    runInEventLoop=cms.untracked.bool(False),
    runInEndLumi=cms.untracked.bool(True),
    runInEndRun=cms.untracked.bool(True),
    runInEndJob=cms.untracked.bool(False)

)


