# Investigatory test for endLuminosityBlock module behavior

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
    u30_infos = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        FwkTest = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('preEventProcessing', 
        'FwkTest'),
    destinations = cms.untracked.vstring('u30_infos')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(12)
)

process.source = cms.Source("EmptySource",
    numberEventsInLuminosityBlock = cms.untracked.uint32(3)
)

process.ssm_1a = cms.EDAnalyzer("UTC_SL1",
    identifier = cms.untracked.int32(11)
)


process.ssm_2a = cms.EDAnalyzer("UTC_SL2",
    identifier = cms.untracked.int32(21)
)


process.ssm_sum = cms.EDAnalyzer("UTC_SLUMMARY"
)

process.p = cms.Path(process.ssm_1a*process.ssm_2a*process.ssm_sum)
