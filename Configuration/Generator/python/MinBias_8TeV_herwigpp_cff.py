import FWCore.ParameterSet.Config as cms

from Configuration.Generator.HerwigppDefaults_cfi import *
from Configuration.Generator.HerwigppUE_V23_cfi import *
from Configuration.Generator.HerwigppPDF_MRST2001_LO_cfi import *
from Configuration.Generator.HerwigppEnergy_8TeV_cfi import *

generator = cms.EDFilter("ThePEGGeneratorFilter",
	herwigDefaultsBlock,
	herwigppUESettingsBlock,
	herwigppPDFSettingsBlock,
	herwigppEnergySettingsBlock,

	configFiles = cms.vstring(),
	parameterSets = cms.vstring(
		'hwpp_cm_8TeV',
		'hwpp_pdf_MRST2001',
		'Summer09QCDParameters',
		'hwpp_cmsDefaults',
	),

	Summer09QCDParameters = cms.vstring(

		'insert /Herwig/MatrixElements/SimpleQCD:MatrixElements[0] /Herwig/MatrixElements/MEMinBias',

		'set /Herwig/Cuts/JetKtCut:MinKT 0.0*GeV',
		'set /Herwig/Cuts/QCDCuts:MHatMin 0.0*GeV',
		'set /Herwig/Cuts/QCDCuts:X1Min 0.01',
		'set /Herwig/Cuts/QCDCuts:X2Min 0.01',
		'set /Herwig/UnderlyingEvent/MPIHandler:IdenticalToUE 0',
	),

	crossSection = cms.untracked.double(101.9e+09),
	filterEfficiency = cms.untracked.double(1.0),
)

ProductionFilterSequence = cms.Sequence(generator)
