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
process.MessageLogger.fwkJobReports = []

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5))

process.source = cms.Source("EmptySource")


from GeneratorInterface.ThePEGInterface.herwigDefaults_cff import *
from GeneratorInterface.ThePEGInterface.herwigValidation_cff import *

process.configurationMetadata = cms.untracked.PSet(
	version = cms.untracked.string('$Revision: 1.5 $'),
	name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/GeneratorInterface/ThePEGInterface/test/testThePEGGeneratorFilter.py,v $'),
	annotation = cms.untracked.string('Herwig++ example - QCD validation')
)

process.generator = cms.EDFilter("ThePEGGeneratorFilter",
	herwigDefaultsBlock,
	herwigValidationBlock,

	configFiles = cms.vstring(
#		'MSSM.model'
	),

	parameterSets = cms.vstring(
		'cmsDefaults', 
#		'validationMSSM',
		'validationQCD'
	),
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('ThePEG.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)
process.schedule = cms.Schedule(process.p, process.outpath)
