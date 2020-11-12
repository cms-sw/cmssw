# u25.cfg Unit test configuration file for MessageLogger service:
# Tests features of concern to users around March 2008, specifically:
# * Behavior of default ERROR limit

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('u25_only'),
    statistics = cms.untracked.vstring('u25_only'),
    categories = cms.untracked.vstring('preEventProcessing', 
        'FwkTest'),
    u25_only = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        FwkTest = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        ERROR = cms.untracked.PSet(
            limit = cms.untracked.int32(3)
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_R")

process.p = cms.Path(process.sendSomeMessages)
