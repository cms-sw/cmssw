import FWCore.ParameterSet.Config as cms

from FWCore.MessageService.MessageLogger_cfi import *
MessageLogger.debugModules = cms.untracked.vstring('')
MessageLogger.cerr =  cms.untracked.PSet(
    FwkReport = cms.untracked.PSet(
        limit = cms.untracked.int32(100),
        reportEvery = cms.untracked.int32(1000)
    ),
    threshold = cms.untracked.string('DEBUG'),
    noLineBreaks = cms.untracked.bool(False),
    DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    Calibration = cms.untracked.PSet(limit = cms.untracked.int32(-1))
)
