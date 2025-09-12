#! /usr/bin/env cmsRun

# This configuration is designed to be run as the second in a series of two cmsRun processes.
# It reads a collection of edmtest::MaybeUninitializedIntProduct from a ROOT file and check their values,
# to test that edm::Wrapper<T> can be used to read and write objects without a default constructor.

import FWCore.ParameterSet.Config as cms

process = cms.Process("Part2")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testMaybeUninitializedIntProduct.root')
)

process.read = cms.EDAnalyzer("edmtest::MaybeUninitializedIntAnalyzer",
    source = cms.InputTag("prod"),
    value = cms.int32(42)
)

process.path = cms.Path(process.read)
