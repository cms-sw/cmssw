import FWCore.ParameterSet.Config as cms

from Configuration.Generator.HerwigppDefaults_cfi import *
from Configuration.Generator.HerwigppUE_V23_cfi import *
from Configuration.Generator.HerwigppPDF_MRST2001_LO_cfi import *
from Configuration.Generator.HerwigppEnergy_7TeV_cfi import *

generator = cms.EDFilter("ThePEGGeneratorFilter",
	herwigDefaultsBlock,
	herwigppUESettingsBlock,
	herwigppPDFSettingsBlock,
	herwigppEnergySettingsBlock,

	configFiles = cms.vstring(),
	parameterSets = cms.vstring(
		'hwpp_cm_7TeV',
		'hwpp_pdf_MRST2001',
		'Summer09QCDParameters',
		'hwpp_cmsDefaults',
	),

	Summer09QCDParameters = cms.vstring(
		'cd /Herwig/MatrixElements/',
		'insert SimpleQCD:MatrixElements[0] MEMinBias',
		'cd /Herwig/Cuts',
		'set JetKtCut:MinKT 0.0*GeV',
		'set QCDCuts:MHatMin 0.0*GeV',
		'set QCDCuts:X1Min 0.01',
		'set QCDCuts:X2Min 0.01',
		'set /Herwig/UnderlyingEvent/MPIHandler:IdenticalToUE 0',
		'cd /',
	),

	crossSection = cms.untracked.double(101.9e+09),
	filterEfficiency = cms.untracked.double(1.0),
)

ProductionFilterSequence = cms.Sequence(generator)
