import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *

##################################################################################################################
### Before Filter: total cross section = 7.893e+10 +- 6.065e+01 pb
### == CMSSW: Filter efficiency (taking into account weights)= (487) / (25000) = 1.948e-02 +- 8.741e-04
### == CMSSW: Filter efficiency (event-level)= (487) / (25000) = 1.948e-02 +- 8.741e-04    [TO BE USED IN MCM]
##################################################################################################################
generator = cms.EDFilter("Pythia8GeneratorFilter",
#                         crossSection = cms.untracked.double(71.39e+09),
                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(1.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(14000.0),
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
            'SoftQCD:inelastic = on'
        ),
        parameterSets = cms.vstring(
            'pythia8CommonSettings',
            'pythia8CP5Settings',
            'processParameters',
        )
    )
)

XiFilter = cms.EDFilter("PythiaFilter",
    MinPt = cms.untracked.double(1.0),
    ParticleID = cms.untracked.int32(3312),
    MaxEta = cms.untracked.double(2.6),
    MinEta = cms.untracked.double(-2.6)
)

ProductionFilterSequence = cms.Sequence(generator*XiFilter)
