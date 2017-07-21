import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tRpctfClient = DQMEDHarvester("L1TRPCTFClient",
input_dir = cms.untracked.string('L1T/L1TRPCTF'),
    prescaleEvt = cms.untracked.int32(1),
    verbose = cms.untracked.bool(False),
    output_dir = cms.untracked.string('L1T/L1TRPCTF/Client'),
    runInEventLoop=cms.untracked.bool(False),
    runInEndLumi=cms.untracked.bool(True),
    runInEndRun=cms.untracked.bool(True),
    runInEndJob=cms.untracked.bool(False)
)


