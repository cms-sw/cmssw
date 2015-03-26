import FWCore.ParameterSet.Config as cms

from Configuration.Generator.HerwigppDefaults_cfi import *


generator = cms.EDFilter("ThePEGGeneratorFilter",
	herwigDefaultsBlock,

	configFiles = cms.vstring(),
	parameterSets = cms.vstring(
		'cm7TeV',
		'pdfMRST2001',
		'Summer09QCDParameters',
		'basicSetup',
		'setParticlesStableForDetector',
	),

	Summer09QCDParameters = cms.vstring(
		'cd /Herwig/MatrixElements/',
		'insert SimpleQCD:MatrixElements[0] MEMinBias',

		'cd /',
		'cd /Herwig/Cuts',
		'set JetKtCut:MinKT 0.0*GeV',
		'set QCDCuts:MHatMin 0.0*GeV',
		'set QCDCuts:X1Min 0.01',
		'set QCDCuts:X2Min 0.01',
		'set /Herwig/UnderlyingEvent/MPIHandler:IdenticalToUE 0',
	),

	crossSection = cms.untracked.double(101.9e+09),
	filterEfficiency = cms.untracked.double(1.0),
)

ProductionFilterSequence = cms.Sequence(generator)
