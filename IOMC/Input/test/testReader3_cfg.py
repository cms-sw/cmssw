#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")

process.source = cms.Source("MCFileSource3",
	fileNames = cms.untracked.vstring('file:GenEvent_ASCII.hepmc3'),
	printEvent = cms.untracked.bool(False)
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))


process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.INFO = cms.untracked.PSet(limit = cms.untracked.int32(-1))

process.GEN = cms.OutputModule("PoolOutputModule",
	fileName = cms.untracked.string('HepMC3_GEN.root')
)

process.outpath = cms.EndPath(process.GEN)
