import FWCore.ParameterSet.Config as cms

rpcEfficiencyHLT = cms.EDAnalyzer("RPCEfficiency",

    incldt = cms.untracked.bool(True),
    incldtMB4 = cms.untracked.bool(True),
    inclcsc = cms.untracked.bool(True),

    debug = cms.untracked.bool(False),
    inves = cms.untracked.bool(True),

    DuplicationCorrection = cms.untracked.int32(1),

    rangestrips = cms.untracked.double(4.),
    rangestripsRB4 = cms.untracked.double(4.),
    MinCosAng = cms.untracked.double(0.99),
    MaxD = cms.untracked.double(80.0),
    MaxDrb4 = cms.untracked.double(150.0),

#    cscSegments = cms.untracked.string('cscSegments'),
#    dt4DSegments = cms.untracked.string('dt4DSegments'),

    cscSegments = cms.untracked.string('hltCscSegments'),
    dt4DSegments = cms.untracked.string('hltDt4DSegments'),


    folderPath = cms.untracked.string('HLT/HLTMonMuon/RPC/'),

    EffSaveRootFile = cms.untracked.bool(False)
)

rpcSourceHLT = cms.Sequence(rpcEfficiencyHLT)



