#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms

process = cms.Process("Writer")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring('file:lhe.root')
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

process.writer = cms.EDAnalyzer("LHEWriter")

process.outpath = cms.EndPath(process.writer)
