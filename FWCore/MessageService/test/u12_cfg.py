# Unit test configuration file for MessageLogger service:
# placeholder feature

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
        u12_warnings = cms.untracked.PSet(
            threshold = cms.untracked.string('WARNING'),
            noTimeStamps = cms.untracked.bool(True)
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_A")

process.p = cms.Path(process.sendSomeMessages)


