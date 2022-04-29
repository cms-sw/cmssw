import FWCore.ParameterSet.Config as cms

DQMStore = cms.Service("DQMStore",
    enableMultiThread = cms.untracked.bool(True),
    saveByLumi = cms.untracked.bool(False),
    trackME = cms.untracked.string(''),
    verbose = cms.untracked.int32(0)
)
