import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

rpcEfficiencySecond = DQMEDHarvester("RPCEfficiencySecond",
    SaveFile = cms.untracked.bool(False),
    NameFile = cms.untracked.string('/tmp/carrillo/RPCEfficiency.root'),
    debug = cms.untracked.bool(False),
)

rpcefficiencysecond = cms.Sequence(rpcEfficiencySecond)


