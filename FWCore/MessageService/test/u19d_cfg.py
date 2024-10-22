# Unit test configuration file for MessageLogger service:
# Intentional long header lines, to test the non-breaking behavior
# Intended to run with code with EDM_ML_DEBUG defined

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(enable = cms.untracked.bool(False)),
    files = cms.untracked.PSet(
      u19d_infos = cms.untracked.PSet(
          threshold = cms.untracked.string('INFO'),
          noTimeStamps = cms.untracked.bool(True),
          FwkTest = cms.untracked.PSet(
              limit = cms.untracked.int32(0)
          ),
          preEventProcessing = cms.untracked.PSet(
              limit = cms.untracked.int32(0)
          )
      ),
      u19d_debugs = cms.untracked.PSet(
          threshold = cms.untracked.string('DEBUG'),
          noTimeStamps = cms.untracked.bool(True),
          FwkTest = cms.untracked.PSet(
              limit = cms.untracked.int32(0)
          ),
          preEventProcessing = cms.untracked.PSet(
              limit = cms.untracked.int32(0)
          )
      )
    ),
    debugModules = cms.untracked.vstring('*')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_Nd")

process.p = cms.Path(process.sendSomeMessages)
