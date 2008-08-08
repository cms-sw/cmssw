import FWCore.ParameterSet.Config as cms

rpcGlobalEfficiency = cms.EDFilter("RPCEfficiencySecond",
    SaveFile = cms.untracked.bool(False),
    NameFile = cms.untracked.string('/tmp/carrillo/RPCEfficiency.root')
)

rpcClient = cms.Sequence(rpcGlobalEfficiency)


