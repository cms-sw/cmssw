import FWCore.ParameterSet.Config as cms

rpcEfficiency = cms.EDFilter("RPCEfficiency",
    cscSegments = cms.untracked.string('cscSegments'),
    dt4DSegments = cms.untracked.string('dt4DSegments'),
    incldtMB4 = cms.untracked.bool(True),
    MaxResStripToCountInAverage = cms.untracked.double(5.0),
    MinimalResidual = cms.untracked.double(4.5),
    inclcsc = cms.untracked.bool(False),
    MinimalResidualRB4 = cms.untracked.double(10.0),
    EffSaveRootFile = cms.untracked.bool(False),
    MaxDrb4 = cms.untracked.double(150.0),
    MinCosAng = cms.untracked.double(0.99),
    muonRPCDigis = cms.untracked.string('muonRPCDigis'),
    EffRootFileName = cms.untracked.string('MuonSegEff.root'),
    MaxResStripToCountInAverageRB4 = cms.untracked.double(7.0),
    incldt = cms.untracked.bool(True),
    EffSaveRootFileEventsInterval = cms.untracked.int32(100),
    MaxD = cms.untracked.double(80.0)
)

rpcSource = cms.Sequence(rpcEfficiency)


