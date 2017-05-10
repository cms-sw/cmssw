import FWCore.ParameterSet.Config as cms

rpcEfficiencySecond = cms.EDAnalyzer("RPCEfficiencySecond",
    SaveFile = cms.untracked.bool(False),
    NameFile = cms.untracked.string('/tmp/carrillo/RPCEfficiency.root'),
    debug = cms.untracked.bool(False),
)

rpcefficiencysecond = cms.Sequence(rpcEfficiencySecond)


