#!/usr/bin/env cmsRun

## Original Author: Andrea Carlo Marini
## Porting to 92X HepMC 2 Gen 
## Date of porting: Mon Jul  3 11:52:22 CEST 2017

import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")

process.source = cms.Source("MCFileSource",
		    #fileNames = cms.untracked.vstring('file:hepmc100.dat'),
			fileNames = cms.untracked.vstring('file:/tmp/amarini/hepmc10K.dat'),
			)
#cmd = "mkfifo /tmp/amarini/hepmc10K.dat"
#cmd = "cat hepmc10K.dat.gz | gunzip -c > /tmp/amarini/hepmc10K.dat"

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))


process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

process.GEN = cms.OutputModule("PoolOutputModule",
		        fileName = cms.untracked.string('HepMC_GEN.root')
			)


process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.genParticles.src= cms.InputTag("source","generator")

process.p = cms.Path(process.genParticles)
process.outpath = cms.EndPath(process.GEN)

### TO DO: add the following
#* merge the following branch (turn off the MT in g4, mv the stuff below from string to input tag)
# amarini/cmssw topic_genevamc 
# add the following line after the sim and digi loading
#process.g4SimHits.HepMCProductLabel = cms.InputTag("source","","GEN")
#process.g4SimHits.Generator.HepMCProductLabel = cms.InputTag("source","","GEN")
#process.genParticles.src=  cms.InputTag("source","","GEN")


### ADD in the different step the following
#
#process.AODSIMoutput.outputCommands.extend([
#		'keep GenRunInfoProduct_*_*_*',
#        	'keep GenLumiInfoProduct_*_*_*',
#		'keep GenEventInfoProduct_*_*_*',
#		])
#
#process.MINIAODSIMoutput.outputcommands.extend([
#       'keep GenRunInfoProduct_*_*_*',
#       'keep GenLumiInfoProduct_*_*_*',
#       'keep GenEventInfoProduct_*_*_*',
#	])
#
# and finally in the ntuples
#process.nero.generator = cms.InputTag("source","generator")
#process.InfoProducer.generator = cms.InputTag("source","generator")
