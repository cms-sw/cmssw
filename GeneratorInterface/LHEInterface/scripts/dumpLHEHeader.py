#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms

from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing ('analysis')

# add a list of strings for events to process
options.register ('input',
				  '',
				  VarParsing.multiplicity.list,
				  VarParsing.varType.string,
				  "Input File")

options.register ('output',
				  'writer.lhe',
				  VarParsing.multiplicity.singleton,
				  VarParsing.varType.string,
				  "Output File")

options.parseArguments()

inFileName = cms.untracked.vstring (options.input)
outFileName = cms.untracked.string (options.output)

process = cms.Process("Writer")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.source = cms.Source("PoolSource",
        processingMode = cms.untracked.string('Runs'),
	fileNames = cms.untracked.vstring(inFileName)
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

process.writer = cms.EDAnalyzer("LHEWriter",
                                output=outFileName)

process.path = cms.Path(process.writer)

process.schedule = cms.Schedule(process.path)
