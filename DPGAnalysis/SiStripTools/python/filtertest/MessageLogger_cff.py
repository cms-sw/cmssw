import FWCore.ParameterSet.Config as cms

from FWCore.MessageLogger.MessageLogger_cfi import MessageLogger

MessageLogger.EventWithHistoryFilterConfiguration = dict()
MessageLogger.files.infos = dict(
    threshold = cms.untracked.string("INFO"),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    FwkReport = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(10000)
    ),
    L1AcceptBunchCrossingNoCollection = cms.untracked.PSet(
        limit = cms.untracked.int32(100)
    )
)

MessageLogger.cerr.threshold = cms.untracked.string("WARNING")


