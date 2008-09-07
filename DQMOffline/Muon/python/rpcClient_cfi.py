import FWCore.ParameterSet.Config as cms

rpcGlobalEfficiency = cms.EDFilter("RPCEfficiencySecond",
    SaveFile = cms.untracked.bool(False),
    NameFile = cms.untracked.string('/tmp/carrillo/RPCEfficiency.root'),
    debug = cms.untracked.bool(False)
)

rpcClient = cms.Sequence(rpcGlobalEfficiency)


