import FWCore.ParameterSet.Config as cms

from Configuration.Generator.HerwigppDefaults_cfi import *
from Configuration.Generator.HerwigppUE_EE_5C_cfi import *
from Configuration.Generator.HerwigppPDF_CTEQ6_LO_cfi import *
from Configuration.Generator.HerwigppEnergy_13TeV_cfi import *

generator = cms.EDFilter("Herwig7GeneratorFilter",
    run = cms.string('InterfaceMatchboxTest'),
    repository = cms.string('${HERWIGPATH}/HerwigDefaults.rpo'),
    dataLocation = cms.string('${HERWIGPATH:-6}'),
    generatorModule = cms.string('/Herwig/Generators/EventGenerator'),
    eventHandlers = cms.string('/Herwig/EventHandlers'),
    configFiles = cms.vstring('LHC-Matchbox.in'),
    crossSection = cms.untracked.double(-1),
    parameterSets = cms.vstring(),
    filterEfficiency = cms.untracked.double(1.0),
)

ProductionFilterSequence = cms.Sequence(generator)
