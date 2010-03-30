# Unit test configuration file for EnableFloatingPointExceptions service

import os # Since we have a general-purpose programming langauge, we'll use it!
import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.Services.InitRootHandlers_cfi")

process.EnableFloatingPointExceptions = cms.Service("EnableFloatingPointExceptions",
    moduleNames = cms.untracked.vstring('default'),
    default = cms.untracked.PSet(
        enableOverFlowEx = cms.untracked.bool(eval(os.getenv("OVERFLOW"))),
        enableDivByZeroEx = cms.untracked.bool(eval(os.getenv("DIVIDEBYZERO"))),
        enableInvalidEx = cms.untracked.bool(eval(os.getenv("INVALID"))),
        enableUnderFlowEx = cms.untracked.bool(eval(os.getenv("UNDERFLOW")))
    ),
    setPrecisionDouble = cms.untracked.bool(True),
    reportSettings = cms.untracked.bool(False)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")
process.module1 = cms.EDAnalyzer("FpeTester", testname = cms.string("overflow"))
process.module2 = cms.EDAnalyzer("FpeTester", testname = cms.string("division"))
process.module3 = cms.EDAnalyzer("FpeTester", testname = cms.string("invalid"))
process.module4 = cms.EDAnalyzer("FpeTester", testname = cms.string("underflow"))
process.p = cms.Path(process.module1*process.module2*process.module3*process.module4)
