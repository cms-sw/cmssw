import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(2)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

from GeneratorInterface.ThePEGInterface.herwigDefaults_cff import *
from GeneratorInterface.ThePEGInterface.herwigValidation_cff import *

process.source = cms.Source("LHESource",
	fileNames = cms.untracked.vstring('file:ttbar.lhe')
)

process.generator = cms.EDFilter("ThePEGHadronizerFilter",
	eventsToPrint = cms.untracked.uint32(1),

	hadronisation = cms.PSet(
		herwigDefaultsBlock,
		herwigValidationBlock,

		generator = cms.string('ThePEG'),

		configFiles = cms.vstring(),

		parameterSets = cms.vstring(
			'pdfCTEQ6LL',
			'basicSetup',
			'cm10TeV',
			'setParticlesStableForDetector',
			'lheDefaults', 
			'lheDefaultPDFs'
		)
	)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('ThePEGHadronizer.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)
process.schedule = cms.Schedule(process.p, process.outpath)

