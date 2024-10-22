# Unit test configuration file for MessageLogger service:
# test message suppression by severity and module (and source)

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    suppressInfo = cms.untracked.vstring('sendSomeMessages'),
    suppressFwkInfo = cms.untracked.vstring('source'),
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        u14_default = cms.untracked.PSet(
            noTimeStamps = cms.untracked.bool(True)
        ),
        u14_errors = cms.untracked.PSet(
            threshold = cms.untracked.string('ERROR'),
            noTimeStamps = cms.untracked.bool(True)
        ),
        u14_infos = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO'),
            noTimeStamps = cms.untracked.bool(True)
        ),
        u14_warnings = cms.untracked.PSet(
            threshold = cms.untracked.string('WARNING'),
            noTimeStamps = cms.untracked.bool(True)
        ),
        u14_debugs = cms.untracked.PSet(
            threshold = cms.untracked.string('DEBUG'),
            noTimeStamps = cms.untracked.bool(True)
        )
    ),
    debugModules = cms.untracked.vstring('*')
)

process.CPU = cms.Service("CPU",
    disableJobReportOutput = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_I")

process.p = cms.Path(process.sendSomeMessages)


