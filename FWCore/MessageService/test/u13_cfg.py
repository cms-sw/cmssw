# Unit test configuration file for MessageLogger service:
# LogVerbatim and LogTrace

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        u13_infos = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO'),
            noTimeStamps = cms.untracked.bool(True),
            FwkReport = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            FwkTest = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            )
        ),
        u13_debugs = cms.untracked.PSet(
            threshold = cms.untracked.string('DEBUG'),
            noTimeStamps = cms.untracked.bool(True),
            FwkReport = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            FwkTest = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            )
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_H")

process.p = cms.Path(process.sendSomeMessages)


