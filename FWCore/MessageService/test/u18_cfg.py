# Unit test configuration file for MessageLogger service:
# LogSystem, LogAbsolute, LogProblem, LogPrint, and LogVerbatim 

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        u18_system = cms.untracked.PSet(
            threshold = cms.untracked.string('ERROR'),
            default = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            noTimeStamps = cms.untracked.bool(True)
        ),
        u18_everything = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO'),
            noTimeStamps = cms.untracked.bool(True),
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_M")

process.p = cms.Path(process.sendSomeMessages)
