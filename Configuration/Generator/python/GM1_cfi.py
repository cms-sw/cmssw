import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *
generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    # put here the efficiency of your filter (1. if no filter)
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    # put here the cross section of your process (in pb)
    crossSection = cms.untracked.double(2.865),
    maxEventsToPrint = cms.untracked.int32(0),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=39                  ! All SUSY processes ', 
            'IMSS(1) = 11             ! Spectrum from external SLHA file', 
            'IMSS(21) = 33            ! LUN number for SLHA File (must be 33) ', 
            'IMSS(22) = 33            ! Read-in SLHA decay table ', 
            'IMSS(11) = 1             ! Set gravitino as LSP', 
            'RMSS(21) = 0             ! The gravitino mass'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters', 
            'SLHAParameters'),
        SLHAParameters = cms.vstring("SLHAFILE = \'Configuration/Generator/data/CSA07SUSYBSM_GM1fran_isasdkpyt_slha.out\'           ! Name of the SLHA spectrum file")
    )
)
