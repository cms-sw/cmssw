# Configuration file for Tracer service

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.Tracer = cms.Service("Tracer")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource")

process.print1 = cms.OutputModule("AsciiOutputModule")

process.print2 = cms.OutputModule("AsciiOutputModule")

process.p = cms.EndPath(process.print1*process.print2)


