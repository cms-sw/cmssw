import FWCore.ParameterSet.Config as cms

from FWCore.MessageLogger.MessageLogger_cfi import MessageLogger

MessageLogger.categories.append("L1AcceptBunchCrossingNoCollection")
MessageLogger.categories.append("EventWithHistoryFilterConfiguration")

MessageLogger.infos.placeholder = cms.untracked.bool(False)
MessageLogger.infos.threshold = cms.untracked.string("INFO")
MessageLogger.infos.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
MessageLogger.infos.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )
MessageLogger.infos.L1AcceptBunchCrossingNoCollection = cms.untracked.PSet(
    limit = cms.untracked.int32(100)
    )
MessageLogger.cerr.L1AcceptBunchCrossingNoCollection = cms.untracked.PSet(
    limit = cms.untracked.int32(100)
    )

MessageLogger.cerr.threshold = cms.untracked.string("WARNING")


