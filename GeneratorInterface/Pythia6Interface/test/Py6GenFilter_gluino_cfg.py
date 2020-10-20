import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.load("SimGeneral.HepPDTESSource.pdt_cfi")


process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)


# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cerr']
process.MessageLogger.statistics = []


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.source = cms.Source("EmptySource")

from Configuration.Generator.PythiaUESettings_cfi import *

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(2),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(10000.0),

    stopHadrons = cms.bool(False),
    gluinoHadrons = cms.bool(True),

    PythiaParameters = cms.PSet(

        pythiaUESettingsBlock,
	#
	# setting for gluino/r-hadrons have been taken from original
	# example application (by F.Moortgat), available in 1_8_4
	# but "lost in translation" from old-style configs to python
	#
        pythiaGluinoRHadrons = cms.vstring(
           'MSEL=0             ! User defined processes',
           'IMSS(1)=1          !  brute force',
           'MSUB(243)=1        !  subprocess',
           'MSUB(244)=1        !  subprocess',
           'IMSS(3)=1',
           'RMSS(3)=300.',
           'RMSS(1)=250.',
           'RMSS(2)=250.',
           'RMSS(4)=10000.',
           'MDCY(309,1)=0      ! set gluino stable',
           'IMSS(5)=1',
           'RMSS(12)=1000.'
           ### 'MSTJ(14)=-1', this is MANDATORY, so it's hardcoded in Py6Had class
           ### 'MSTP(111)=0'  this is MANDATORY, so it's hardcoded in Py6Had class
	),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring( 
	    'pythiaUESettings', 
            'pythiaGluinoRHadrons')
    )
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('gluinoRHadrons.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
