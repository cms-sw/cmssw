import FWCore.ParameterSet.Config as cms
from Configuration.GenProduction.PythiaUESettings_cfi import *

source = cms.Source("LHESource",
	fileNames = cms.untracked.vstring('file:events.lhe'),
	seekEvent = cms.untracked.uint32(0)
)

generator = cms.EDProducer("LHEProducer",
	hadronisation = cms.PSet(
		pythiaUESettingsBlock,
		generator = cms.string('Pythia6'),
		parameterSets = cms.vstring(
			'pythiaUESettings'
		)
	)
)

ProducerFilterSequence = cms.Sequence(generator)
