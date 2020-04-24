import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

rpcGlobalEfficiencyHLT = DQMEDHarvester("RPCEfficiencySecond",
    SaveFile = cms.untracked.bool(False),
    debug = cms.untracked.bool(False),
    barrel = cms.untracked.bool(True),
    endcap = cms.untracked.bool(True),
    folderPath = cms.untracked.string('HLT/HLTMonMuon/RPC/')
)

rpcClientHLT = cms.Sequence(rpcGlobalEfficiencyHLT)
