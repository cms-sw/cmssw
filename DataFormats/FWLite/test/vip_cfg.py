# Configuration file for testing vector of built-in type

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.VIP = cms.EDProducer("IntVectorProducer",
    ivalue = cms.int32(42),
    count = cms.int32(1)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('vectorinttest.root')
)

process.p = cms.Path(process.VIP)
process.outp = cms.EndPath(process.out)
