# u28_cerr_cfg.py:
#
# Non-regression test configuration file for MessageLogger service:
# distinct threshold level for linked destination, where
#

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        statisticsThreshold = cms.untracked.string('WARNING'),
        enableStatistics = cms.untracked.bool(True)
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_A")

process.p = cms.Path(process.sendSomeMessages)
