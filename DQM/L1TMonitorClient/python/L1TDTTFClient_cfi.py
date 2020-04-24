import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tDttfClient = DQMEDHarvester("L1TDTTFClient",
    l1tSourceFolder = cms.untracked.string('L1T/L1TDTTF'),
    dttfSource = cms.InputTag("l1tDttf"),
    online = cms.untracked.bool(True),
    resetAfterLumi = cms.untracked.int32(3)
)


