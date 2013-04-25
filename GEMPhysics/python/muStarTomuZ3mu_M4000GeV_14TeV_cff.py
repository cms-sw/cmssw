import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUEZ2starSettings_cfi import *
generator = cms.EDFilter("Pythia8GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(14000.0),
    crossSection = cms.untracked.double(0.13074),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring(
	    'Tune:pp 5', 
            'ExcitedFermion:qqbar2muStarmu = on', 
            'ExcitedFermion:Lambda= 10000', 
            '4000013:onMode = off', 
            '4000013:onIfMatch = 13 23', 
            '4000013:m0 = 4000', 
            '23:onMode = off', 
            '23:onIfMatch = 13 13'),
        parameterSets = cms.vstring('processParameters')
    )
)

