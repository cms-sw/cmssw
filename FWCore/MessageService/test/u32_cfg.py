# Unit test configuration file for MessageLogger service
# Uses include MessageLogger.cfi, and overrides destinations to just errors.
#  Used to investigate spoor log fiels like infos.log
#  Derived from u9_cfg.py

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.errors = cms.untracked.PSet(
    threshold = cms.untracked.string('ERROR'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(3)
    ),
    noTimeStamps = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_G")

process.p = cms.Path(process.sendSomeMessages)
