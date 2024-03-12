#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms

process = cms.Process("LHE")

process.source = cms.Source("AlpgenSource",
                            fileNames = cms.untracked.vstring('file:w2j')
                            )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.load("FWCore.MessageService.MessageLogger_cfi")

process.writer = cms.EDAnalyzer("LHEWriter")

process.load("Configuration.EventContent.EventContent_cff")

process.LHE = cms.OutputModule("PoolOutputModule",
                                       process.FEVTSIMEventContent,
                                       fileName = cms.untracked.string('testSource.root')
                              )

process.outpath = cms.EndPath(process.writer + process.LHE)
# foo bar baz
# EDr4AYyHrInAX
# FKGL6L4JKkZ5S
