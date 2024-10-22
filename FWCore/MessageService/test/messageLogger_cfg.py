# Configuration file for MessageLogger service

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('sendSomeMessages'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(100),
        timespan = cms.untracked.int32(60)
    ),
    files = cms.untracked.PSet(
        critical = cms.untracked.PSet(
            threshold = cms.untracked.string('ERROR')
        ),
        detailedInfo = cms.untracked.PSet(
            INFO = cms.untracked.PSet(
                timespan = cms.untracked.int32(1000)
            ),
            default = cms.untracked.PSet(
                limit = cms.untracked.int32(10),
                timespan = cms.untracked.int32(600)
            ),
            threshold = cms.untracked.string('DEBUG'),
            trkwarning = cms.untracked.PSet(
                limit = cms.untracked.int32(100),
                timespan = cms.untracked.int32(30)
            ),
            unimportant = cms.untracked.PSet(
                limit = cms.untracked.int32(5)
            )
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("MessageLoggerClient")

process.p = cms.Path(process.sendSomeMessages)
