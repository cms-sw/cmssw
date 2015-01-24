## Wprime to muon, Z2* tune, 50x production in winter 2012


import FWCore.ParameterSet.Config as cms



from Configuration.Generator.PythiaUEZ2starSettings_cfi import *
generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.),
    crossSection = cms.untracked.double(0.02123),
    comEnergy = cms.double(13000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL        = 0    !User defined processes', 
                                        'MSUB(142)   = 1    !Wprime  production',
                                        'PMAS(34,1)  = 2000.!mass of Wprime',
                                        'MDME(311,1) = 0    !W\' decay into dbar u', 
                                        'MDME(312,1) = 0    !W\' decay into dbar c', 
                                        'MDME(313,1) = 0    !W\' decay into dbar t', 
                                        'MDME(315,1) = 0    !W\' decay into sbar u', 
                                        'MDME(316,1) = 0    !W\' decay into sbar c', 
                                        'MDME(317,1) = 0    !W\' decay into sbar t', 
                                        'MDME(319,1) = 0    !W\' decay into bbar u', 
                                        'MDME(320,1) = 0    !W\' decay into bbar c', 
                                        'MDME(321,1) = 0    !W\' decay into bbar t', 
                                        'MDME(327,1) = 0    !W\' decay into e+ nu_e', 
                                        'MDME(328,1) = 1    !W\' decay into mu+ nu_mu', 
                                        'MDME(329,1) = 0    !W\' decay into tau+ nu_tau'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)




ProductionFilterSequence = cms.Sequence(generator)
