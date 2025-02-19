import FWCore.ParameterSet.Config as cms
from GeneratorInterface.ThePEGInterface.herwigDefaults_cff import *
from GeneratorInterface.ThePEGInterface.herwigValidation_cff import *

configurationMetadata = cms.untracked.PSet(
	version = cms.untracked.string('$Revision: 1.5 $'),
	name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/GeneratorInterface/ThePEGInterface/test/testThePEGHadronisation.py,v $'),
	annotation = cms.untracked.string('LHE example - ttbar events')
)

source = cms.Source("LHESource",
	fileNames = cms.untracked.vstring('file:ttbar.lhe')
)

generator = cms.EDProducer("LHEProducer",
	eventsToPrint = cms.untracked.uint32(1),

	hadronisation = cms.PSet(
		herwigDefaultsBlock,
		herwigValidationBlock,

		generator = cms.string('ThePEG'),

		configFiles = cms.vstring(),

		parameterSets = cms.vstring(
			'pdfCTEQ5L',
			'basicSetup',
			'cm10TeV',
			'setParticlesStableForDetector',
			'lheDefaults', 
			'lheDefaultPDFs'
		)
	)
)
