import FWCore.ParameterSet.Config as cms

from Configuration.Generator.HerwigppDefaults_cfi import *
from Configuration.Generator.HerwigppUE_EE_5C_cfi import *
from Configuration.Generator.HerwigppPDF_CTEQ6_LO_cfi import *
from Configuration.Generator.HerwigppEnergy_13TeV_cfi import *

generator = cms.EDFilter("ThePEGGeneratorFilter",
        herwigDefaultsBlock,
	herwigppUESettingsBlock,
	herwigppPDFSettingsBlock,
	herwigppEnergySettingsBlock,

        configFiles = cms.vstring(),
        parameterSets = cms.vstring(
                'hwpp_cmsDefaults',
                'hwpp_cm_13TeV',
                'hwpp_ue_EE5C',
		'hwpp_pdf_CTEQ6L1',
                'processParameters',
        ),

        processParameters = cms.vstring(

                'insert /Herwig/MatrixElements/SimpleQCD:MatrixElements[0] /Herwig/MatrixElements/MEQCD2to2',

                'set /Herwig/Cuts/JetKtCut:MinKT 30*GeV',
                'set /Herwig/Cuts/QCDCuts:MHatMin 0.0*GeV',
                'set /Herwig/UnderlyingEvent/MPIHandler:IdenticalToUE 0',
        ),

        crossSection = cms.untracked.double(-1.),
        filterEfficiency = cms.untracked.double(1.0),
)

ProductionFilterSequence = cms.Sequence(generator)
