# Unit test configuration file for MessageLogger service:
#   By-severity limit on a type of message, 
#   and specific-category override of that default

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.MessageLogger = cms.Service("MessageLogger",
    default = cms.untracked.PSet(
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    u21_infos = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('preEventProcessing', 
        'FwkJob', 
        'importantInfo', 
        'routineInfo'),
    u21_warnings = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FWKINFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(True),
        importantInfo = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('u21_warnings', 
        'u21_infos')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_O")

process.p = cms.Path(process.sendSomeMessages)
