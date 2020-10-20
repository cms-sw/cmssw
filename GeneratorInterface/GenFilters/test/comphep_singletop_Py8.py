# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

#from Configuration.Generator.PythiaUEZ2starSettings_cfi import *
#from Configuration.GenProduction.PythiaUESettings_cfi import *

process = cms.Process("TEST")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService", 
	generator = cms.PSet(initialSeed = cms.untracked.uint32(123456789), 
	engineName = cms.untracked.string('HepJamesRandom') ))

# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")
##process.MessageLogger.destinations = ['cerr']
##process.MessageLogger.statistics = []
##
##process.MessageLogger.cerr.threshold = "Warning"


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.source = cms.Source("LHESource",
        fileNames = cms.untracked.vstring('file:')#file name 
)

import FWCore.ParameterSet.Config as cms

process.generator = cms.EDFilter("Pythia8HadronizerFilter",
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(2),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    comEnergy = cms.double(13000.),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring(
		'Main:timesAllowErrors    = 10000', 
        'ParticleDecays:limitTau0 = on',
		'ParticleDecays:tauMax = 10',
        'Tune:ee 3',
        'Tune:pp 5'
        ),
        parameterSets = cms.vstring('processParameters')
    )
)

process.selection = cms.EDFilter("ComphepSingletopFilterPy8",
    pTSep = cms.double(99999999), # 99999999 - tq, 0 -tqb
)


process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('matching_comphep_singletop.root'), 
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p')),
   # outputCommands = cms.untracked.vstring('drop *') # !!! Drop all events, if you do not need matching_comphep_singletop.root file
)

process.p = cms.Path(process.generator * process.selection)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
