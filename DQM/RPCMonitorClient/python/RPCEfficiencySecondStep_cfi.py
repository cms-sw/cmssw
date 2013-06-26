import FWCore.ParameterSet.Config as cms

rpcEfficiencySecond = cms.EDAnalyzer("RPCEfficiencySecond",
    SaveFile = cms.untracked.bool(False),
    NameFile = cms.untracked.string('/tmp/carrillo/RPCEfficiency.root'),
    debug = cms.untracked.bool(False),
    barrel = cms.untracked.bool(True),
    endcap = cms.untracked.bool(True)
)

rpcefficiencysecond = cms.Sequence(rpcEfficiencySecond)


