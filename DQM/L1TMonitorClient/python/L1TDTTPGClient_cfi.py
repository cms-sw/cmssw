import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tDttpgClient = DQMEDHarvester("L1TDTTPGClient",
    input_dir = cms.untracked.string('L1T/L1TDTTPG'),
    prescaleLS = cms.untracked.int32(-1),
    output_dir = cms.untracked.string('L1T/L1TDTTPG/Tests'),
    prescaleEvt = cms.untracked.int32(500)
)


