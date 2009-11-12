import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *

generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(10),
    # Print event
    pythiaPylistVerbosity = cms.untracked.int32(3),
    # Print decay tables
#    pythiaPylistVerbosity = cms.untracked.int32(12),                         
    filterEfficiency = cms.untracked.double(1.0),
    comEnergy = cms.double(10000.0),
#    crossSection = cms.untracked.double(55000000000.),
    UseExternalGenerators = cms.untracked.bool(False),
#
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        pythiaMyParameters = cms.vstring(
# This is needed if you are introducing long-lived exotic particles.
          'MSTJ(22)=1    ! Decay ALL unstable particles',

          'MSEL=0',
# Request gg -> H0 production
          'MSUB(152)=1',
          'MWID(35)=2 ! Let me set H0 properties'
        ),
#
        PYUPDAParameters = cms.vstring(
# Read my parameters
         "PYUPDAFILE = 'Configuration/Generator/data/Pythia_H0_pyupda.in'"
# Optionally call PYUPDA after PYINIT. This doesn't seem to be necessary.
#         "PYUPDApostPYINIT"
# Write current parameters
#         "PYUPDAFILE = \'pyupda.out\'"
#         "PYUPDAWRITE"
        ),
#
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
                                    'pythiaMyParameters',
                                    'PYUPDAParameters')
    )
)

# this needs to get in somehow...
#
# genParticles.abortOnUnknownPDGCode = cms.untracked.bool(False)


