# Unit test configuration file for messageSummaryToJobReport
#   Tests proper output of both ungrouped and grouped categories
#   in summary placed in the FrameworkJobReport.xml. 
#   Suggested regression-test usage (.sh file): 
#   cmsRun -j u27FJR.xml

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    messageSummaryToJobReport = cms.untracked.bool(True),
    default = cms.untracked.PSet(
        FwkTest = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        u27_infos = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO'),
            noTimeStamps = cms.untracked.bool(True),
            enableStatistics = cms.untracked.bool(True),
            FwkTest = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            )
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.ssm_1a = cms.EDAnalyzer("UTC_Q1",
    identifier = cms.untracked.int32(11)
)

process.ssm_1b = cms.EDAnalyzer("UTC_Q1",
    identifier = cms.untracked.int32(12)
)

process.ssm_1c = cms.EDAnalyzer("UTC_Q1",
    identifier = cms.untracked.int32(13)
)

process.ssm_2a = cms.EDAnalyzer("UTC_Q2",
    identifier = cms.untracked.int32(21)
)

process.ssm_2b = cms.EDAnalyzer("UTC_Q2",
    identifier = cms.untracked.int32(22)
)

process.ssm_2c = cms.EDAnalyzer("UTC_Q2",
    identifier = cms.untracked.int32(23)
)

process.p = cms.Path(process.ssm_1a*process.ssm_2a*process.ssm_1b*process.ssm_2b*process.ssm_1c*process.ssm_2c)
