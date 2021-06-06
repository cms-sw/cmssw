# timing_t.cfg
# NON-REGRESSSION Unit test configuration file for MessageLogger service:
# This variant puts Timing into job report.  
# Tester should run with FrameworkJobReport.fwk and timing_t.log for proper timing info.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.Timing = cms.Service("Timing",
    useJobReport = cms.untracked.bool(True)
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('*'),
    default = cms.untracked.PSet(
        FwkTest = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    files = cms.untracked.PSet(
        timing_t = cms.untracked.PSet(
            noTimeStamps = cms.untracked.bool(True),
            threshold = cms.untracked.string('ERROR')
        )
    ),
    u16_job_report = cms.untracked.PSet(
        extension = cms.untracked.string('mmxml')
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_A")

process.p = cms.Path(process.sendSomeMessages)
