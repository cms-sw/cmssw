import FWCore.ParameterSet.Config as cms

from Configuration.Generator.HerwigppDefaults_cfi import *
from Configuration.Generator.HerwigppUE_EE_5C_cfi import *
from Configuration.Generator.HerwigppPDF_CTEQ6_LO_cfi import *
from Configuration.Generator.HerwigppEnergy_13TeV_cfi import *
from Configuration.Generator.HerwigppMECorrections_cfi import *
from Configuration.Generator.HerwigppReshuffle_RestMostOffShell_cfi import *

generator = cms.EDFilter("ThePEGGeneratorFilter",
	herwigDefaultsBlock,
	herwigppUESettingsBlock,
	herwigppPDFSettingsBlock,
	herwigppEnergySettingsBlock,
	herwigppMECorrectionsSettingsBlock,
	herwigppReshuffleSettingsBlock,

	configFiles = cms.vstring(),
	parameterSets = cms.vstring(
		'hwpp_cmsDefaults',
		'hwpp_ue_EE5C',
		'hwpp_pdf_CTEQ6L1',
		'hwpp_cm_13TeV',
		'hwpp_MECorr_Off',
		'hwpp_reshuffle_RestMostOffShell',
		'ttbarprocess',

	),

	ttbarprocess = cms.vstring(
                'insert /Herwig/MatrixElements/SimpleQCD:MatrixElements[0] /Herwig/MatrixElements/MEHiggs',
		'insert /Herwig/MatrixElements/SimpleQCD:MatrixElements[0] /Herwig/MatrixElements/MEHiggsJet',
		'set /Herwig/MatrixElements/MEHiggsJet:Process gg',
		'set /Herwig/Cuts/JetKtCut:MinKT 0.2*GeV', # Standard recommendation by Herwig++ authors 0 GeV
	),

        crossSection = cms.untracked.double(-1),
        filterEfficiency = cms.untracked.double(1.0),
)
ProductionFilterSequence = cms.Sequence(generator)
