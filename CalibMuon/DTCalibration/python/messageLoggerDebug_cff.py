import FWCore.ParameterSet.Config as cms

from FWCore.MessageService.MessageLogger_cfi import *
MessageLogger.debugModules = cms.untracked.vstring('')
MessageLogger.destinations = cms.untracked.vstring('cerr')
MessageLogger.categories.append('Calibration')
MessageLogger.cerr =  cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'),
    noLineBreaks = cms.untracked.bool(False),
    DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    Calibration = cms.untracked.PSet(limit = cms.untracked.int32(-1))
)
