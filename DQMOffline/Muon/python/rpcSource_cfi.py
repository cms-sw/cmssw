import FWCore.ParameterSet.Config as cms

rpcEfficiency = cms.EDFilter("RPCEfficiency",
    incldt = cms.untracked.bool(True),
    incldtMB4 = cms.untracked.bool(True),
    inclcsc = cms.untracked.bool(True),

    debug = cms.untracked.bool(False),

    DuplicationCorrection = cms.untracked.int32(1),

    MinimalResidual = cms.untracked.double(4.5),
    MinimalResidualRB4 = cms.untracked.double(10.0),
    MinCosAng = cms.untracked.double(0.99),
    MaxD = cms.untracked.double(80.0),
    MaxDrb4 = cms.untracked.double(150.0),
    MaxResStripToCountInAverage = cms.untracked.double(5.0),
    MaxResStripToCountInAverageRB4 = cms.untracked.double(7.0),

    muonRPCDigis = cms.untracked.string('muonRPCDigis'),
    cscSegments = cms.untracked.string('cscSegments'),
    dt4DSegments = cms.untracked.string('dt4DSegments'),

    EffSaveRootFile = cms.untracked.bool(False),
    EffRootFileName = cms.untracked.string('MuonSegEff.root'),
    EffSaveRootFileEventsInterval = cms.untracked.int32(100)

)

rpcSource = cms.Sequence(rpcEfficiency)


