import FWCore.ParameterSet.Config as cms

rpcEfficiency = cms.EDFilter("RPCEfficiency",
    incldt = cms.untracked.bool(True),
    incldtMB4 = cms.untracked.bool(True),
    inclcsc = cms.untracked.bool(True),

    debug = cms.untracked.bool(False),
<<<<<<< rpcSource_cfi.py
    inves = cms.untracked.bool(True),
=======
    inves = cms.untracked.bool(False),
>>>>>>> 1.4

    DuplicationCorrection = cms.untracked.int32(1),

    rangestrips = cms.untracked.double(1.),
    rangestripsRB4 = cms.untracked.double(4.),
    MinCosAng = cms.untracked.double(0.99),
    MaxD = cms.untracked.double(80.0),
    MaxDrb4 = cms.untracked.double(150.0),

    muonRPCDigis = cms.untracked.string('muonRPCDigis'),
    cscSegments = cms.untracked.string('cscSegments'),
    dt4DSegments = cms.untracked.string('dt4DSegments'),

<<<<<<< rpcSource_cfi.py
    EffSaveRootFile = cms.untracked.bool(True),
    EffRootFileName = cms.untracked.string('/tmp/carrillo/RPCEfficiencyFIRST.root'),
=======
    EffSaveRootFile = cms.untracked.bool(False),
    EffRootFileName = cms.untracked.string('output.root'),
>>>>>>> 1.4
    EffSaveRootFileEventsInterval = cms.untracked.int32(100)
)

rpcSource = cms.Sequence(rpcEfficiency)


