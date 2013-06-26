# Configuration file for MessageLogger service

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    messageIDs = cms.untracked.vstring('unimportant', 
        'trkwarning'),
    anotherfile = cms.untracked.PSet(
        postBeginJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(100),
        timespan = cms.untracked.int32(60)
    ),
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        INFO = cms.untracked.PSet(
            timespan = cms.untracked.int32(1000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10),
            timespan = cms.untracked.int32(600)
        ),
        unimportant = cms.untracked.PSet(
            limit = cms.untracked.int32(5)
        ),
        trkwarning = cms.untracked.PSet(
            limit = cms.untracked.int32(100),
            timespan = cms.untracked.int32(30)
        )
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    fwkJobReports = cms.untracked.vstring('anotherfile'),
    debugModules = cms.untracked.vstring('sendSomeMessages'),
    categories = cms.untracked.vstring('postBeginJob'),
    destinations = cms.untracked.vstring('detailedInfo', 
        'critical')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("MessageLoggerClient")

process.p = cms.Path(process.sendSomeMessages)
