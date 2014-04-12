import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUEZ2starSettings_cfi import *
generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    crossSection = cms.untracked.double(17120.0),
    comEnergy = cms.double(13000.0),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL        = 0    !User defined processes', 
            'MSUB(2)     = 1    !W production', 
            'MDME(190,1) = 0    !W decay into dbar u', 
            'MDME(191,1) = 0    !W decay into dbar c', 
            'MDME(192,1) = 0    !W decay into dbar t', 
            'MDME(194,1) = 0    !W decay into sbar u', 
            'MDME(195,1) = 0    !W decay into sbar c', 
            'MDME(196,1) = 0    !W decay into sbar t', 
            'MDME(198,1) = 0    !W decay into bbar u', 
            'MDME(199,1) = 0    !W decay into bbar c', 
            'MDME(200,1) = 0    !W decay into bbar t', 
            'MDME(206,1) = 0    !W decay into e+ nu_e', 
            'MDME(207,1) = 1    !W decay into mu+ nu_mu', 
            'MDME(208,1) = 0    !W decay into tau+ nu_tau'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)
