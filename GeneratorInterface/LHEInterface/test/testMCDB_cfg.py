#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms

process = cms.Process("LHE")

process.source = cms.Source("MCDBSource",
	articleID = cms.uint32(186),
	supportedProtocols = cms.untracked.vstring('rfio')
	#filter = cms.untracked.string('\\.lhe$')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.configurationMetadata = cms.untracked.PSet(
	version = cms.untracked.string('alpha'),
	name = cms.untracked.string('LHEF input'),
	annotation = cms.untracked.string('ttbar')
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

process.LHE = cms.OutputModule("PoolOutputModule",
	dataset = cms.untracked.PSet(dataTier = cms.untracked.string('LHE')),
	fileName = cms.untracked.string('lhe.root')
)

process.outpath = cms.EndPath(process.LHE)
