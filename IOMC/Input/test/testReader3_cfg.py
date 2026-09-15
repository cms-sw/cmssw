#!/usr/bin/env cmsRun
import os
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.setDefault('inputFiles', 'file:' + os.path.join(os.environ.get('SCRAM_TEST_PATH', '.'), 'UnpGenEvent10.hepmc'))
options.setDefault('outputFile', 'HepMC3_GEN.root')
options.setDefault('maxEvents', -1)
options.register('printEvent',
                 False,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 'Print the content of every event which is read.')
options.parseArguments()

process = cms.Process("GEN")

process.source = cms.Source("MCFileSource3",
	fileNames = cms.untracked.vstring(options.inputFiles),
	printEvent = cms.untracked.bool(options.printEvent)
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))


process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.INFO = cms.untracked.PSet(limit = cms.untracked.int32(-1))

process.GEN = cms.OutputModule("PoolOutputModule",
	fileName = cms.untracked.string(options.outputFile)
)

process.outpath = cms.EndPath(process.GEN)
