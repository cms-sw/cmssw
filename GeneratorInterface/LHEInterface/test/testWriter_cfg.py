#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms

process = cms.Process("Writer")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring('file:lhe.root')
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

process.writer = cms.EDAnalyzer("LHEWriter")

process.outpath = cms.EndPath(process.writer)
# foo bar baz
# ye7v74iA78V9f
# 45wSjvqYb2pbF
