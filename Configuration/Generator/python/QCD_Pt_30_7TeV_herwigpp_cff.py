import FWCore.ParameterSet.Config as cms

from Configuration.Generator.HerwigppDefaults_cfi import *


generator = cms.EDFilter("ThePEGGeneratorFilter",
	herwigDefaultsBlock,

	configFiles = cms.vstring(),
	parameterSets = cms.vstring(
		'hwpp_cm_7TeV',
		'hwpp_pdf_MRST2001',
		'Summer09QCDParameters',
		'hwpp_cmsDefaults',
	),

	Summer09QCDParameters = cms.vstring(
		'cd /Herwig/MatrixElements/',
		'insert SimpleQCD:MatrixElements[0] MEQCD2to2',

		'set /Herwig/Cuts/JetKtCut:MinKT 30*GeV',
		'set /Herwig/Cuts/QCDCuts:MHatMin 0.0*GeV',
		'set /Herwig/UnderlyingEvent/MPIHandler:IdenticalToUE 0',
		'cd /',
	),

	crossSection = cms.untracked.double(6.22927e+07),
	filterEfficiency = cms.untracked.double(1.0),
)

ProductionFilterSequence = cms.Sequence(generator)
