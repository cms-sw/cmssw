import FWCore.ParameterSet.Config as cms
from GeneratorInterface.ThePEGInterface.herwigDefaults_cff import *

configurationMetadata = cms.untracked.PSet(
	version = cms.untracked.string('$Revision: 1.6 $'),
	name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/GenProduction/python/Herwigpp_base_cff.py,v $'),
	annotation = cms.untracked.string('LHE example - ttbar events, MRST2001 used, MinKT=1400 GeV')
)

source = cms.Source("LHESource",
	fileNames = cms.untracked.vstring('file:ttbar.lhe')
)

generator = cms.EDProducer("LHEProducer",
	eventsToPrint = cms.untracked.uint32(1),
	dumpConfig  = cms.untracked.string(""),
	dumpEvents  = cms.untracked.string(""),

	hadronisation = cms.PSet(
		herwigDefaultsBlock,

		generator = cms.string('ThePEG'),

		configFiles = cms.vstring(),

		parameterSets = cms.vstring(
			'cmsDefaults', 
			'lheDefaults', 
			'lheDefaultPDFs'
		)
	)
)

ProducerSourceSequence = cms.Sequence(generator)
