import FWCore.ParameterSet.Config as cms

MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailledInfo'),
    debugModules = cms.untracked.vstring('*'),
    categories = cms.untracked.vstring('CkfPattern'),
    detailledInfo = cms.untracked.PSet(
        INFO = cms.untracked.PSet(    limit = cms.untracked.int32(0) ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),

        CkfPattern = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),

        default = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
        Root_NoDictionary = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        FwkSummary = cms.untracked.PSet(reportEvery = cms.untracked.int32(1),limit = cms.untracked.int32(10000000) ),
        threshold = cms.untracked.string('DEBUG')
    )
)



