# Test of a feature of PSet validation:
#   messageSummaryToJobReport

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",

#enable one of the following -- the first four should pass, the rest fail
#    threshold = cms.untracked.string('DEBUG'),
#  threshold = cms.untracked.string('INFO'),
#    threshold = cms.untracked.string('WARNING'),
   threshold = cms.untracked.string('ERROR'),
#    threshold = cms.string('ERROR'),
#    threshold = cms.untracked.int32('ERROR'),
#    threshold = cms.untracked.string('error'),

#to test multiple stringing of flaws, enable this:
#    messageSummaryToJobReport = cms.untracked.int32(2),

    u1_infos = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    u1_warnings = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noTimeStamps = cms.untracked.bool(True)
    ),
    u1_debugs = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        noTimeStamps = cms.untracked.bool(True),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    u1_default = cms.untracked.PSet(
        noTimeStamps = cms.untracked.bool(True),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    u1_errors = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        noTimeStamps = cms.untracked.bool(True)
    ),
    fwkJobReports = cms.untracked.vstring('u1_job_report.mxml'),
    debugModules = cms.untracked.vstring('*'),
    categories = cms.untracked.vstring('preEventProcessing', 
        'FwkJob'),
    destinations = cms.untracked.vstring('u1_warnings', 
        'u1_errors', 
        'u1_infos', 
        'u1_debugs', 
        'u1_default')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_A")

process.p = cms.Path(process.sendSomeMessages)
