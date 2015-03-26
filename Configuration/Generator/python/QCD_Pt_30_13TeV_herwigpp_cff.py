import FWCore.ParameterSet.Config as cms

from Configuration.Generator.HerwigppDefaults_cfi import *
from Configuration.Generator.HerwigppUE_EE_5C_cfi import *

generator = cms.EDFilter("ThePEGGeneratorFilter",
        herwigDefaultsBlock,
        herwigppUESettingsBlock,
        configFiles = cms.vstring(),
        processParameters = cms.vstring(
                'cd /Herwig/MatrixElements/',
                'insert SimpleQCD:MatrixElements[0] MEQCD2to2',

                'cd /',
                'set /Herwig/Cuts/JetKtCut:MinKT 30*GeV',
                'set /Herwig/Cuts/QCDCuts:MHatMin 0.0*GeV',
                'set /Herwig/UnderlyingEvent/MPIHandler:IdenticalToUE 0',
        ),
        parameterSets = cms.vstring(
                'cmsDefaults',
                'cm13TeV',
                'EE5C',
                'processParameters',
        ),
        crossSection = cms.untracked.double(-1.),
        filterEfficiency = cms.untracked.double(1.0),
)

ProductionFilterSequence = cms.Sequence(generator)
