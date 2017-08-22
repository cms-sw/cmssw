import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
    )
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(2)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5))

process.source = cms.Source("EmptySource")

from GeneratorInterface.ThePEGInterface.herwigValidation_cff import *
process.load('Configuration.Generator.HerwigppDefaults_cfi')

process.generator = cms.EDFilter("ThePEGGeneratorFilter",
	process.herwigDefaultsBlock,
	herwigValidationBlock,

	configFiles = cms.vstring(
#		'MSSM.model'
	),
	cmsDefaults = cms.vstring(),

	parameterSets = cms.vstring(
		'cmsDefaults', 
#		'validationMSSM',
		'validationQCD'
	),
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('ThePEGGenerator.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)
process.schedule = cms.Schedule(process.p, process.outpath)
