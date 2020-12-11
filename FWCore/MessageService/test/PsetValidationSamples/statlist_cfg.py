# Test of a feature of PSet validation:
#   The  statistics

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('*'),
    files = cms.untracked.PSet(
        u1_debugs = cms.untracked.PSet(
            FwkTest = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            noTimeStamps = cms.untracked.bool(True),
            preEventProcessing = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            threshold = cms.untracked.string('DEBUG')
        ),
        u1_default = cms.untracked.PSet(
            FwkTest = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            enableStatistics = cms.untracked.bool(True),
            noTimeStamps = cms.untracked.bool(True),
            preEventProcessing = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            )
        ),
        u1_errors = cms.untracked.PSet(
            noTimeStamps = cms.untracked.bool(True),
            threshold = cms.untracked.string('ERROR')
        ),
        u1_infos = cms.untracked.PSet(
            FwkTest = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            noTimeStamps = cms.untracked.bool(True),
            preEventProcessing = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            threshold = cms.untracked.string('INFO')
        ),
        u1_warnings = cms.untracked.PSet(
            enableStatistics = cms.untracked.bool(True),
            noTimeStamps = cms.untracked.bool(True),
            threshold = cms.untracked.string('WARNING')
        ),
        u1_x = cms.untracked.PSet(

        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_A")

process.p = cms.Path(process.sendSomeMessages)
