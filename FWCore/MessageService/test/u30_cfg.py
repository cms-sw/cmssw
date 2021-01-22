# Unit test configuration file for GroupLogStatistics(category)
#   Tests LoggedErrorsSummary, using modues that have normal categories
#   and also grouped categories.   The expectation is that even grouped
#   categories will be separated by module on the per-event summary.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.MessageLogger = cms.Service("MessageLogger",
    default = cms.untracked.PSet(
        FwkTest = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        )
    ),
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        u30_infos = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO'),
            noTimeStamps = cms.untracked.bool(True),
            FwkTest = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            )
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource")

process.ssm_1a = cms.EDAnalyzer("UTC_S1",
    identifier = cms.untracked.int32(11)
)


process.ssm_2a = cms.EDAnalyzer("UTC_S2",
    identifier = cms.untracked.int32(21)
)


process.ssm_sum = cms.EDAnalyzer("UTC_SUMMARY"
)

process.p = cms.Path(process.ssm_1a*process.ssm_2a*process.ssm_sum)
