# Unit test configuration file for MessageLogger service:
# threshold levels for destinations
# limit=0 for a category (needed to avoid time stamps in files to be compared)
# enabling all (*) LogDebug, with one destination responding
# verify that by default, the threshold for a destination is INFO
# test done with debug enabled

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
      u1d_infos = cms.untracked.PSet(
          threshold = cms.untracked.string('INFO'),
          noTimeStamps = cms.untracked.bool(True),
          FwkTest = cms.untracked.PSet(
              limit = cms.untracked.int32(0)
          ),
          preEventProcessing = cms.untracked.PSet(
              limit = cms.untracked.int32(0)
          )
      ),
      u1d_warnings = cms.untracked.PSet(
          threshold = cms.untracked.string('WARNING'),
          noTimeStamps = cms.untracked.bool(True)
      ),
      u1d_debugs = cms.untracked.PSet(
          threshold = cms.untracked.string('DEBUG'),
          noTimeStamps = cms.untracked.bool(True),
          FwkTest = cms.untracked.PSet(
              limit = cms.untracked.int32(0)
          ),
          preEventProcessing = cms.untracked.PSet(
              limit = cms.untracked.int32(0)
          )
      ),
      u1d_default = cms.untracked.PSet(
          noTimeStamps = cms.untracked.bool(True),
          FwkTest = cms.untracked.PSet(
              limit = cms.untracked.int32(0)
          ),
          preEventProcessing = cms.untracked.PSet(
              limit = cms.untracked.int32(0)
          )
      ),
      u1d_errors = cms.untracked.PSet(
          threshold = cms.untracked.string('ERROR'),
          noTimeStamps = cms.untracked.bool(True)
      )
    ),
    debugModules = cms.untracked.vstring('*')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_Ad")

process.p = cms.Path(process.sendSomeMessages)
