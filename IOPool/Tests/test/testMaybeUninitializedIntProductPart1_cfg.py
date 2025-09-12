#! /usr/bin/env cmsRun

# This configuration is designed to be run as the first in a series of two cmsRun processes.
# It produces a collection of edmtest::MaybeUninitializedIntProduct and stores them to a ROOT file,
# to test that edm::Wrapper<T> can be used to read and write objects without a default constructor.

import FWCore.ParameterSet.Config as cms

process = cms.Process("Part1")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.prod = cms.EDProducer("edmtest::MaybeUninitializedIntProducer",
    value = cms.int32(42)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testMaybeUninitializedIntProduct.root'),
    outputCommands = cms.untracked.vstring(
        'keep *_prod_*_*'
    )
)

process.path = cms.Path(process.prod)
process.endp = cms.EndPath(process.out)
